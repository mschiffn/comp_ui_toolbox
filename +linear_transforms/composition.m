%
% composition of linear transforms (chain operation)
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2019-08-10
%
classdef composition < linear_transforms.linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        transforms

        % dependent properties
        N_transforms

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = composition( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % multiple varargin{ 1 } / single varargin{ index_arg }
            for index_arg = 2:nargin
                if ~isscalar( varargin{ 1 } ) && isscalar( varargin{ index_arg } )
                    varargin{ index_arg } = repmat( varargin{ index_arg }, size( varargin{ 1 } ) );
                end
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            % check individual transforms for validity and size
            N_coefficients = cell( size( varargin ) );
            N_points = cell( size( varargin ) );

            % iterate arguments
            for index_arg = 1:nargin

                % ensure class linear_transforms.linear_transform
                if ~isa( varargin{ index_arg }, 'linear_transforms.linear_transform' )
                    errorStruct.message = sprintf( 'varargin{ %d } must be linear_transforms.linear_transform!', index_arg );
                    errorStruct.identifier = 'composition:NoLinearTransforms';
                    error( errorStruct );
                end

                % get sizes of linear transforms
                N_coefficients{ index_arg } = reshape( [ varargin{ index_arg }.N_coefficients ], size( varargin{ index_arg } ) );
                N_points{ index_arg } = reshape( [ varargin{ index_arg }.N_points ], size( varargin{ index_arg } ) );

                % check compatible sizes of linear transforms for composition
                if index_arg ~= 1

                    % iterate linear transforms
                    for index_transform = 1:numel( varargin{ index_arg } )

                        if N_points{ index_arg - 1 }( index_transform ) ~= N_coefficients{ index_arg }( index_transform )
                            errorStruct.message = sprintf( 'N_points{ %d }( %d ) must equal N_coefficients{ %d }( %d )!', index_arg - 1, index_transform, index_arg, index_transform );
                            errorStruct.identifier = 'composition:IncompatibleSize';
                            error( errorStruct );
                        end

                    end % for index_transform = 1:numel( varargin{ index_arg } )

                end % if index_arg ~= 1

            end % for index_arg = 1:nargin

            %--------------------------------------------------------------
            % 2.) create compositions of linear transforms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.linear_transform( N_coefficients{ 1 }, N_points{ end } );

            % iterate compositions of linear transforms
            for index_object = 1:numel( varargin{ 1 } )

                % set independent properties
                objects( index_object ).transforms = cell( nargin, 1 );
                for index_transform = 1:nargin
                    objects( index_object ).transforms{ index_transform } = varargin{ index_transform }( index_object );
                end

                % set dependent properties
                objects( index_object ).N_transforms = nargin;
%                 objects( index_object ).size_transforms = size;

            end % for index_object = 1:numel( varargin{ 1 } )

        end % function objects = composition( varargin )

        %------------------------------------------------------------------
        % forward transform (overload forward_transform method)
        %------------------------------------------------------------------
        function y = forward_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.composition
            if ~isa( LTs, 'linear_transforms.composition' )
                errorStruct.message = 'LTs must be linear_transforms.composition!';
                errorStruct.identifier = 'forward_transform:NoCompositions';
                error( errorStruct );
            end

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
                LTs = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute forward transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate compositions of linear transforms
            for index_object = 1:numel( LTs )

                % compute first forward transform
                y_temp = LTs( index_object ).transforms{ LTs( index_object ).N_transforms }.forward_transform( x );

                % compose remaining forward transforms
                for index_transform = ( LTs( index_object ).N_transforms - 1 ):-1:2

                    y_temp = LTs( index_object ).transforms{ index_transform }.forward_transform( y_temp );

                end % for index_transform = ( LTs( index_object ).N_transforms - 1 ):-1:2

                % compute last forward transform
                y{ index_object } = LTs( index_object ).transforms{ 1 }.forward_transform( y_temp );

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single composition of linear transforms
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform (overload adjoint_transform method)
        %------------------------------------------------------------------
        function y = adjoint_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.composition
            if ~isa( LTs, 'linear_transforms.composition' )
                errorStruct.message = 'LTs must be linear_transforms.composition!';
                errorStruct.identifier = 'adjoint_transform:NoCompositions';
                error( errorStruct );
            end

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
                LTs = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute adjoint transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate compositions of linear transforms
            for index_object = 1:numel( LTs )

                % compute first adjoint transform
                y_temp = LTs( index_object ).transforms{ 1 }.adjoint_transform( x );

                % compose remaining adjoint transforms
                for index_transform = 2:( LTs( index_object ).N_transforms - 1 )

                    y_temp = LTs( index_object ).transforms{ index_transform }.adjoint_transform( y_temp );

                end % for index_transform = 2:( LTs( index_object ).N_transforms - 1 )

                % compute last adjoint transform
                y{ index_object } = LTs( index_object ).transforms{ LTs( index_object ).N_transforms }.adjoint_transform( y_temp );

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single composition of linear transforms
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = adjoint_transform( LTs, x )

        %------------------------------------------------------------------
        % threshold
        %------------------------------------------------------------------
        function [ LTs, N_threshold ] = threshold( LTs, xis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.weighting
            if ~isa( LTs, 'linear_transforms.weighting' )
                errorStruct.message = 'LTs must be linear_transforms.weighting!';
                errorStruct.identifier = 'threshold:NoWeighting';
                error( errorStruct );
            end

            % ensure valid xis ( 0; 1 ]
            mustBePositive( xis );
            mustBeLessThanOrEqual( xis, 1 );

            % multiple LTs / single xis
            if ~isscalar( LTs ) && isscalar( xis )
                xis = repmat( xis, size( LTs ) );
            end

            % single LTs / multiple xis
            if isscalar( LTs ) && ~isscalar( xis )
                LTs = repmat( LTs, size( xis ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, xis );

            %--------------------------------------------------------------
            % 2.) apply thresholds to diagonal weighting matrices
            %--------------------------------------------------------------
            % initialize N_threshold with zeros
            N_threshold = zeros( size( LTs ) );

            % iterate diagonal weighting matrices
            for index_object = 1:numel( LTs )

                % compute threshold
                one_over_lb = min( LTs( index_object ).weights ) / xis( index_object );

                % detect invalid weights
                indicator = LTs( index_object ).weights > one_over_lb;
                N_threshold( index_object ) = sum( indicator );

                % apply threshold
                LTs( index_object ).weights( indicator ) = one_over_lb;
                LTs( index_object ).weights_conj = conj( LTs( index_object ).weights );

            end % for index_object = 1:numel( LTs )

        end % function LTs = threshold( LTs, xis )

    end % methods

end % classdef composition < linear_transforms.linear_transform
