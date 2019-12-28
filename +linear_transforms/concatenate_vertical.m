%
% superclass for all vertical concatenations of linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-10
% modified: 2019-12-18
%
classdef concatenate_vertical < linear_transforms.linear_transform

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        transforms

        % dependent properties
        N_transforms ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 2  % number of concatenated linear transforms

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = concatenate_vertical( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.linear_transform
            for index_arg = 1:numel( varargin )
                if ~isa( varargin{ index_arg }, 'linear_transforms.linear_transform' )
                    errorStruct.message = sprintf( 'varargin{ %d } must be linear_transforms.linear_transform!', index_arg );
                    errorStruct.identifier = 'concatenate_vertical:NoLinearTransforms';
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
                errorStruct.identifier = 'concatenate_vertical:NoLinearTransforms';
                error( errorStruct );
            end

            % constructor of superclass
            objects@linear_transforms.linear_transform( N_coefficients, N_points( :, 1 ) );

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

        end % function objects = concatenate_vertical( varargin )

        %------------------------------------------------------------------
        % forward transform (implement forward_transform method)
        %------------------------------------------------------------------
        function y = forward_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.concatenate_vertical
            if ~isa( LTs, 'linear_transforms.concatenate_vertical' )
                errorStruct.message = 'LTs must be linear_transforms.concatenate_vertical!';
                errorStruct.identifier = 'forward_transform:NoVerticalConcatenations';
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

            % iterate vertical concatenations
            for index_object = 1:numel( LTs )

                y{ index_object } = cell( LTs( index_object ).N_transforms, 1 );

                % apply vertically concatenated forward transforms
                for index_transform = 1:LTs( index_object ).N_transforms

                    y{ index_object }{ index_transform } = forward_transform( LTs( index_object ).transforms{ index_transform }, x{ index_object } );

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
            % ensure class linear_transforms.concatenate_vertical
            if ~isa( LTs, 'linear_transforms.concatenate_vertical' )
                errorStruct.message = 'LTs must be linear_transforms.concatenate_vertical!';
                errorStruct.identifier = 'forward_transform:NoVerticalConcatenations';
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

            % iterate vertical concatenations
            for index_object = 1:numel( LTs )

                % initialize y{ index_object } w/ zeros
                y{ index_object } = zeros( LTs( index_object ).N_points, 1 );

                % partition input
                N_coefficients = cellfun( @( x ) x.N_coefficients, LTs( index_object ).transforms );
                x{ index_object } = mat2cell( x{ index_object }, N_coefficients, size( x{ index_object }, 2 ) );

                % apply vertically concatenated adjoint transforms
                for index_transform = 1:LTs( index_object ).N_transforms

                    y{ index_object } = y{ index_object } + adjoint_transform( LTs( index_object ).transforms{ index_transform }, x{ index_object }{ index_transform } );

                end % for index_transform = 1:LTs( index_object ).N_transforms

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = adjoint_transform( LTs, x )

    end % methods
    
end % classdef concatenate_vertical < linear_transforms.linear_transform