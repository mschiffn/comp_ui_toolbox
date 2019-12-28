%
% compute identity for various options
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2019-11-21
%
classdef identity < linear_transforms.orthonormal_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = identity( N_points )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures nonempty positive integers for N_points

            %--------------------------------------------------------------
            % 2.) create identity operators
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.orthonormal_linear_transform( N_points );

        end % function objects = identity( N_points )

        %------------------------------------------------------------------
        % forward transform (overload forward_transform method)
        %------------------------------------------------------------------
        function y = forward_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
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
            % 2.) compute identity operators
            %--------------------------------------------------------------
            % copy input cell array
            y = x;

            % iterate identity operators
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform (overload adjoint_transform method)
        %------------------------------------------------------------------
        function y = adjoint_transform( LTs, x )

            % adjoint transform equals forward transform
            y = forward_transform( LTs, x );

        end % function y = adjoint_transform( LTs, x )

        %------------------------------------------------------------------
        % threshold (TODO: inherit from weighting)
        %------------------------------------------------------------------
        function [ LTs, N_threshold ] = threshold( LTs, xis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.identity
            if ~isa( LTs, 'linear_transforms.identity' )
                errorStruct.message = 'LTs must be linear_transforms.identity!';
                errorStruct.identifier = 'threshold:NoIdentity';
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
            % 2.) apply thresholds to identity transforms
            %--------------------------------------------------------------
            % initialize N_threshold with zeros
            N_threshold = zeros( size( LTs ) );

        end % function [ LTs, N_threshold ] = threshold( LTs, xis )

    end % methods

end % classdef identity < linear_transforms.orthonormal_linear_transform
