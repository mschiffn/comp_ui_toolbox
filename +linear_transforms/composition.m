%
% composition of linear transforms (chain operation)
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-04-16
%
classdef composition < linear_transforms.linear_transform_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        transforms

        % dependent properties
        N_transforms ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 1  % number of composed linear transforms

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = composition( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure valid number of input arguments
            narginchk( 2, inf );

            % ensure classes linear_transforms.linear_transform
            indicator = cellfun( @( x ) ~isa( x, 'linear_transforms.linear_transform' ), varargin );
            if any( indicator( : ) )
                errorStruct.message = 'All arguments must be linear_transforms.linear_transform!';
                errorStruct.identifier = 'composition:NoLinearTransforms';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ varargin{ : } ] = auxiliary.ensureEqualSize( varargin{ : } );

            % detect identity transforms
            indicator_identity = cellfun( @( x ) isa( x, 'linear_transforms.identity' ), varargin );

%             % detect compositions of linear transforms
%             indicator_composition = cellfun( @( x ) isa( x, 'linear_transforms.composition' ), varargin );

            % check individual transforms for validity and size
            N_coefficients = cell( size( varargin ) );
            N_points = cell( size( varargin ) );

            % iterate arguments
            for index_arg = 1:nargin

                % get sizes of linear transforms
                N_coefficients{ index_arg } = reshape( [ varargin{ index_arg }.N_coefficients ], size( varargin{ index_arg } ) );
                N_points{ index_arg } = reshape( [ varargin{ index_arg }.N_points ], size( varargin{ index_arg } ) );

                % check compatible sizes of linear transforms for composition
                if index_arg ~= 1

                    % iterate linear transforms
                    for index_transform = 1:numel( varargin{ index_arg } )

                        if N_points{ index_arg - 1 }( index_transform ) ~= N_coefficients{ index_arg }( index_transform )
                            errorStruct.message = sprintf( 'N_points{ %d }( %d ) must equal N_coefficients{ %d }( %d )!', index_arg - 1, index_transform, index_arg, index_transform );
                            errorStruct.identifier = 'composition:SizeMismatch';
                            error( errorStruct );
                        end

                    end % for index_transform = 1:numel( varargin{ index_arg } )

                end % if index_arg ~= 1

            end % for index_arg = 1:nargin

            %--------------------------------------------------------------
            % 2.) create compositions of linear transforms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_coefficients{ 1 }, N_points{ end } );

            % iterate compositions of linear transforms
            for index_object = 1:numel( varargin{ 1 } )

                % set independent properties
                objects( index_object ).transforms = cell( nargin, 1 );
                for index_transform = 1:nargin
                    objects( index_object ).transforms{ index_transform } = varargin{ index_transform }( index_object );
                end

                % decompose compositions

                % remove identities from composition
                objects( index_object ).transforms( indicator_identity ) = [];

                % set dependent properties
                objects( index_object ).N_transforms = numel( objects( index_object ).transforms );
%                 objects( index_object ).size_transforms = size;

            end % for index_object = 1:numel( varargin{ 1 } )

        end % function objects = composition( varargin )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single matrix)
        %------------------------------------------------------------------
        function y = forward_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.composition (scalar)
            if ~( isa( LT, 'linear_transforms.composition' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.composition!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleComposition';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward compositions (single matrix)
            %--------------------------------------------------------------
            % compute first forward transform
            y_temp = forward_transform( LT.transforms{ end }, x );

            % compose remaining forward transforms
            for index_transform = ( LT.N_transforms - 1 ):-1:2

                y_temp = forward_transform( LT.transforms{ index_transform }, y_temp );

            end % for index_transform = ( LT.N_transforms - 1 ):-1:2

            % compute last forward transform
            y = forward_transform( LT.transforms{ 1 }, y_temp );

        end % function y = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function y = adjoint_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.composition (scalar)
            if ~( isa( LT, 'linear_transforms.composition' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.composition!';
                errorStruct.identifier = 'adjoint_transform_matrix:NoSingleComposition';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint convolutions (single matrix)
            %--------------------------------------------------------------
            % compute first adjoint transform
            y_temp = adjoint_transform( LT.transforms{ 1 }, x );

            % compose remaining adjoint transforms
            for index_transform = 2:( LT.N_transforms - 1 )

                y_temp = adjoint_transform( LT.transforms{ index_transform }, y_temp );

            end % for index_transform = 2:( LT.N_transforms - 1 )

            % compute last adjoint transform
            y = adjoint_transform( LT.transforms{ end }, y_temp );

        end % function y = adjoint_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single matrix)
        %------------------------------------------------------------------
        function display_coefficients_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.composition (scalar)
            if ~( isa( LT, 'linear_transforms.composition' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.composition!';
                errorStruct.identifier = 'display_coefficients_matrix:NoSingleComposition';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) display coefficients (single matrix)
            %--------------------------------------------------------------
            % display coefficients of last forward transform
            display_coefficients_matrix( LT.transforms{ end }, x );

        end % function display_coefficients_matrix( LT, x )

        %------------------------------------------------------------------
        % relative RMSEs of best s-sparse approximations (single matrix)
        %------------------------------------------------------------------
        function [ rel_RMSEs, axes_s ] = rel_RMSE_matrix( LT, y )
        end

	end % methods (Access = protected, Hidden)

end % classdef composition < linear_transforms.linear_transform_matrix
