%
% superclass for all diagonal concatenations of linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2019-12-17
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
            % ensure class linear_transforms.linear_transform
            for index_arg = 1:numel( varargin )
                if ~isa( varargin{ index_arg }, 'linear_transforms.linear_transform' )
                    errorStruct.message = sprintf( 'varargin{ %d } must be linear_transforms.linear_transform!', index_arg );
                    errorStruct.identifier = 'concatenate_diagonal:NoLinearTransforms';
                    error( errorStruct );
                end
            end % for index_arg = 1:numel( varargin )

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create diagonal concatenations
            %--------------------------------------------------------------
            % number of concatenated linear transforms
            N_transforms = nargin;

            % specify cell arrays
            N_coefficients = cell( N_transforms, 1 );
            N_points = cell( N_transforms, 1 );

            % iterate arguments
            for index_arg = 1:N_transforms

                % extract numbers of coefficients and points
                N_coefficients{ index_arg } = reshape( [ varargin{ index_arg }.N_coefficients ], [ 1, numel( varargin{ index_arg } ) ] );
                N_points{ index_arg } = reshape( [ varargin{ index_arg }.N_points ], [ 1, numel( varargin{ index_arg } ) ] );

            end % for index_arg = 1:N_transforms

            % concatenate vertically
            N_coefficients = cat( 1, N_coefficients{ : } );
            N_points = cat( 1, N_points{ : } );

            % constructor of superclass
            objects@linear_transforms.linear_transform( sum( N_coefficients, 1 ), sum( N_points, 1 ) );

            % iterate vertical concatenations
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).transforms = cell( N_transforms, 1 );
                for index_arg = 1:N_transforms
                    objects( index_object ).transforms{ index_arg } = varargin{ index_arg }( index_object );
                end

                % set dependent properties
                objects( index_object ).N_transforms = N_transforms;
                objects( index_object ).sizes = [ N_coefficients( :, index_object ), N_points( :, index_object ) ];

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
