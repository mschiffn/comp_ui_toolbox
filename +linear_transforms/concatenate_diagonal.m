%
% superclass for all diagonal concatenations of linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-01-06
%
classdef concatenate_diagonal < linear_transforms.linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        transforms

        % dependent properties
        N_transforms ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 1  % number of concatenated linear transforms
        sizes ( :, 2 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 1, 1 ]	% sizes of the concatenated linear transforms

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = concatenate_diagonal( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure sufficient number of arguments
            if nargin < 2
                errorStruct.message = 'A diagonal concatenation requires at least two linear transforms!';
                errorStruct.identifier = 'concatenate_diagonal:InsufficientNumberOfLinearTransforms';
                error( errorStruct );
            end

            % ensure classes linear_transforms.linear_transform
            indicator = cellfun( @( x ) ~isa( x, 'linear_transforms.linear_transform' ), varargin );
            if any( indicator( : ) )
                errorStruct.message = 'All arguments must be linear_transforms.linear_transform!';
                errorStruct.identifier = 'concatenate_diagonal:NoLinearTransforms';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create diagonal concatenations
            %--------------------------------------------------------------
% TODO: detect weightings and identities
            % detect diagonal concatenations
            indicator_diagonal = cellfun( @( x ) isa( x, 'linear_transforms.concatenate_diagonal' ), varargin );

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
            objects@linear_transforms.linear_transform( sum( N_coefficients, 1 ), sum( N_points, 1 ) );

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

        end % function objects = concatenate_diagonal( varargin )

        %------------------------------------------------------------------
        % forward transform (implement forward_transform method)
        %------------------------------------------------------------------
        function y = forward_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.concatenate_diagonal
            if ~isa( LTs, 'linear_transforms.concatenate_diagonal' )
                errorStruct.message = 'LTs must be linear_transforms.concatenate_diagonal!';
                errorStruct.identifier = 'forward_transform:NoDiagonalConcatenations';
                error( errorStruct );
            end

            % LTs ensure numeric matrices for x

            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                x = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute forward transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate diagonal concatenations of linear transforms
            for index_object = 1:numel( LTs )

                % partition input matrix
                x{ index_object } = mat2cell( x{ index_object }, LTs( index_object ).sizes( :, 2 ), size( x{ index_object }, 2 ) );

                % specify cell array for y{ index_object }
                y{ index_object } = cell( LTs( index_object ).N_transforms, 1 );

                % apply individual forward transforms
                for index_transform = 1:LTs( index_object ).N_transforms

                    y{ index_object }{ index_transform } = forward_transform( LTs( index_object ).transforms{ index_transform }, x{ index_object }{ index_transform } );

                end % for index_transform = 1:LTs( index_object ).N_transforms

                % concatenate vertically
                y{ index_object } = cat( 1, y{ index_object }{ : } );

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform (implement adjoint_transform method)
        %------------------------------------------------------------------
        function y = adjoint_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.concatenate_diagonal
            if ~isa( LTs, 'linear_transforms.concatenate_diagonal' )
                errorStruct.message = 'LTs must be linear_transforms.concatenate_diagonal!';
                errorStruct.identifier = 'adjoint_transform:NoDiagonalConcatenations';
                error( errorStruct );
            end

            % LTs ensure numeric matrices for x

            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                x = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute adjoint transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate diagonal concatenations of linear transforms
            for index_object = 1:numel( LTs )

                % partition input matrix
                x{ index_object } = mat2cell( x{ index_object }, LTs( index_object ).sizes( :, 1 ), size( x{ index_object }, 2 ) );

                % specify cell array for y{ index_object }
                y{ index_object } = cell( LTs( index_object ).N_transforms, 1 );

                % apply individual adjoint transforms
                for index_transform = 1:LTs( index_object ).N_transforms

                    y{ index_object }{ index_transform } = adjoint_transform( LTs( index_object ).transforms{ index_transform }, x{ index_object }{ index_transform } );

                end % for index_transform = 1:LTs( index_object ).N_transforms

                % concatenate vertically
                y{ index_object } = cat( 1, y{ index_object }{ : } );

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = adjoint_transform( LTs, x )

    end % methods

end % classdef concatenate_diagonal < linear_transforms.linear_transform
