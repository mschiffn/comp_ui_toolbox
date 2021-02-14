%
% superclass for all vertical concatenations of linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-10
% modified: 2020-10-26
%
% TODO: < linear_transforms.concatenations.concatenation
classdef vertical < linear_transforms.linear_transform_matrix

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        transforms

        % dependent properties
        N_transforms ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 2  % number of concatenated linear transforms
% TODO: sizes
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = vertical( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.linear_transform
            for index_arg = 1:numel( varargin )
                if ~isa( varargin{ index_arg }, 'linear_transforms.linear_transform' )
                    errorStruct.message = sprintf( 'varargin{ %d } must be linear_transforms.linear_transform!', index_arg );
                    errorStruct.identifier = 'vertical:NoLinearTransforms';
                    error( errorStruct );
                end
            end % for index_arg = 1:numel( varargin )

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create vertical concatenations
            %--------------------------------------------------------------
            % specify cell arrays
            N_coefficients = cell( 1, numel( varargin ) );
            N_points = cell( 1, numel( varargin ) );

            % iterate arguments
            for index_arg = 1:numel( varargin )

                % extract numbers of coefficients and points
                N_coefficients{ index_arg } = reshape( [ varargin{ index_arg }.N_coefficients ], [ numel( varargin{ index_arg } ), 1 ] );
                N_points{ index_arg } = reshape( [ varargin{ index_arg }.N_points ], [ numel( varargin{ index_arg } ), 1 ] );

            end % for index_arg = 1:numel( varargin )

            N_coefficients = sum( cat( 2, N_coefficients{ : } ), 2 );
            N_points = cat( 2, N_points{ : } );

            % ensure equal numbers of points
            indicator = abs( N_points - N_points( :, 1 ) ) > eps;
            if any( indicator( : ) )
                errorStruct.message = 'Linear transforms to concatenate must have the same numbers of points!';
                errorStruct.identifier = 'vertical:NoLinearTransforms';
                error( errorStruct );
            end

            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_coefficients, N_points( :, 1 ) );

            % iterate vertical concatenations
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).transforms = cell( nargin, 1 );
                for index_arg = 1:numel( varargin )
                    objects( index_object ).transforms{ index_arg } = varargin{ index_arg }( index_object );
                end

                % set dependent properties
                objects( index_object ).N_transforms = nargin;

            end % for index_object = 1:numel( objects )

        end % function objects = vertical( varargin )

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
            % ensure class linear_transforms.concatenations.vertical (scalar)
            if ~( isa( LT, 'linear_transforms.concatenations.vertical' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.concatenations.vertical!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleVerticalConcatenation';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward vertical concatenations (single matrix)
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( LT.N_transforms, 1 );

            % apply vertically concatenated forward transforms
            for index_transform = 1:LT.N_transforms

                y{ index_transform } = forward_transform( LT.transforms{ index_transform }, x );

            end % for index_transform = 1:LT.N_transforms

            % concatenate vertically
            y = cat( 1, y{ : } );

        end % function y = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function y = adjoint_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.concatenations.vertical (scalar)
            if ~( isa( LT, 'linear_transforms.concatenations.vertical' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.concatenations.vertical!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleVerticalConcatenation';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint vertical concatenations (single matrix)
            %--------------------------------------------------------------
            % initialize y w/ zeros
            y = zeros( LT.N_points, 1 );

            % partition input
            N_coefficients = cellfun( @( x ) x.N_coefficients, LT.transforms );
            x = mat2cell( x, N_coefficients, size( x, 2 ) );

            % apply vertically concatenated adjoint transforms
            for index_transform = 1:LT.N_transforms

                y = y + adjoint_transform( LT.transforms{ index_transform }, x{ index_transform } );

            end % for index_transform = 1:LT.N_transforms

        end % function y = adjoint_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single matrix)
        %------------------------------------------------------------------
        function display_coefficients_matrix( LT, x )

            % partition input
            N_coefficients = cellfun( @( x ) x.N_coefficients, LT.transforms );
            x = mat2cell( x, N_coefficients, size( x, 2 ) );

            % iterate linear transforms
            for index_transform = 1:LT.N_transforms

                % display coefficients of individual transforms
                subplot( LT.N_transforms, 1, index_transform );
                display_coefficients_matrix( LT.transforms{ index_transform }, x{ index_transform } );

            end % for index_transform = 1:LT.N_transforms

        end % function display_coefficients_matrix( LT, x )

        %------------------------------------------------------------------
        % relative RMSEs of best s-sparse approximations (single matrix)
        %------------------------------------------------------------------
        function [ rel_RMSEs, axes_s ] = rel_RMSE_matrix( LT, y )
        end

	end % methods (Access = protected, Hidden)

end % classdef vertical < linear_transforms.linear_transform_matrix
