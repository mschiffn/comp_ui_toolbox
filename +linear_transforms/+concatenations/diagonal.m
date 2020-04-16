%
% superclass for all diagonal concatenations of linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-04-16
%
% TODO: < linear_transforms.concatenations.concatenation
classdef diagonal < linear_transforms.linear_transform_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        transforms

        % dependent properties
        N_transforms ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 1  % number of concatenated linear transforms
        sizes ( :, 2 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 1, 1 ]	% sizes of the concatenated linear transforms

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = diagonal( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure valid number of input arguments
            narginchk( 2, inf );

            % ensure classes linear_transforms.linear_transform
            indicator = cellfun( @( x ) ~isa( x, 'linear_transforms.linear_transform' ), varargin );
            if any( indicator( : ) )
                errorStruct.message = 'All arguments must be linear_transforms.linear_transform!';
                errorStruct.identifier = 'diagonal:NoLinearTransforms';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ varargin{ : } ] = auxiliary.ensureEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create diagonal concatenations
            %--------------------------------------------------------------
% TODO: detect weightings and identities
            % detect diagonal concatenations
            indicator_diagonal = cellfun( @( x ) isa( x, 'linear_transforms.concatenations.diagonal' ), varargin );

            % numbers of concatenated linear transforms
            N_transforms = ones( 1, numel( varargin{ 1 } ), nargin );
            for index_arg = 1:nargin
                if indicator_diagonal( index_arg )
                    N_transforms( 1, :, index_arg ) = [ varargin{ index_arg }.N_transforms ];
                end
            end
            N_transforms_sum = sum( N_transforms, 3 );

            % specify cell arrays
            N_coefficients = cell( nargin, 1 );
            N_points = cell( nargin, 1 );

            % iterate arguments
            for index_arg = 1:nargin

                % extract numbers of coefficients and points
                N_coefficients{ index_arg } = reshape( [ varargin{ index_arg }.N_coefficients ], [ 1, numel( varargin{ index_arg } ) ] );
                N_points{ index_arg } = reshape( [ varargin{ index_arg }.N_points ], [ 1, numel( varargin{ index_arg } ) ] );

            end % for index_arg = 1:nargin

            % concatenate vertically
            N_coefficients = cat( 1, N_coefficients{ : } );
            N_points = cat( 1, N_points{ : } );

            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( sum( N_coefficients, 1 ), sum( N_points, 1 ) );

            % reshape diagonal concatenations
            objects = reshape( objects, size( varargin{ 1 } ) );

            % iterate diagonal concatenations
            for index_object = 1:numel( objects )

                % set independent properties
                indices = mat2cell( 1:N_transforms_sum( index_object ), 1, N_transforms( 1, index_object, : ) );
                objects( index_object ).transforms = cell( N_transforms_sum( index_object ), 1 );
                objects( index_object ).sizes = ones( N_transforms_sum( index_object ), 2 );
                for index_arg = 1:nargin

                    if indicator_diagonal( index_arg )
                        objects( index_object ).transforms( indices{ index_arg } ) = varargin{ index_arg }( index_object ).transforms;
                    else
                        objects( index_object ).transforms{ indices{ index_arg } } = varargin{ index_arg }( index_object );
                    end

                end % for index_arg = 1:nargin

                % set dependent properties
                objects( index_object ).N_transforms = N_transforms_sum( index_object );
                objects( index_object ).sizes = [ cellfun( @( x ) x.N_coefficients, objects( index_object ).transforms ), cellfun( @( x ) x.N_points, objects( index_object ).transforms ) ];

            end % for index_object = 1:numel( objects )

        end % function objects = diagonal( varargin )

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
            % ensure class linear_transforms.concatenations.diagonal (scalar)
            if ~( isa( LT, 'linear_transforms.concatenations.diagonal' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.concatenations.diagonal!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleDiagonalConcatenation';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward diagonal concatenations (single matrix)
            %--------------------------------------------------------------
            % partition input matrix
            x = mat2cell( x, LT.sizes( :, 2 ), size( x, 2 ) );

            % specify cell array for y
            y = cell( LT.N_transforms, 1 );

            % apply individual forward transforms
            for index_transform = 1:LT.N_transforms

                y{ index_transform } = forward_transform( LT.transforms{ index_transform }, x{ index_transform } );

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
            % ensure class linear_transforms.concatenations.diagonal (scalar)
            if ~( isa( LT, 'linear_transforms.concatenations.diagonal' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.concatenations.diagonal!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleDiagonalConcatenation';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint diagonal concatenations (single matrix)
            %--------------------------------------------------------------
            % partition input matrix
            x = mat2cell( x, LT.sizes( :, 1 ), size( x, 2 ) );

            % specify cell array for y
            y = cell( LT.N_transforms, 1 );

            % apply individual adjoint transforms
            for index_transform = 1:LT.N_transforms

                y{ index_transform } = adjoint_transform( LT.transforms{ index_transform }, x{ index_transform } );

            end % for index_transform = 1:LT.N_transforms

            % concatenate vertically
            y = cat( 1, y{ : } );

        end % function y = adjoint_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single matrix)
        %------------------------------------------------------------------
        function display_coefficients_matrix( LT, x )

        end % function display_coefficients_matrix( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef diagonal < linear_transforms.linear_transform_matrix
