%
% compute effect of diagonal weighting matrix
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2019-05-20
%
classdef weighting < linear_transforms.invertible_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        weights ( :, 1 )

        % dependent properties
        weights_conj ( :, 1 )

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = weighting( weights )
% TODO: threshold!
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for weights
            if ~iscell( weights )
                weights = { weights };
            end

            %--------------------------------------------------------------
            % 2.) create diagonal weighting matrices
            %--------------------------------------------------------------
            % numbers of weights
            N_weights = cellfun( @numel, weights );

            % constructor of superclass
            objects@linear_transforms.invertible_linear_transform( N_weights );

            % iterate diagonal weighting matrices
            for index_object = 1:numel( weights )

                % ensure numeric weights
                if ~isnumeric( weights{ index_object } )
                    errorStruct.message = sprintf( 'weights{ %d } must be numeric!', index_object );
                    errorStruct.identifier = 'weighting:InvalidWeights';
                    error( errorStruct );
                end

                % ensure nonzero weights
                if any( double( abs( weights{ index_object }( : ) ) ) < eps )
                    errorStruct.message = sprintf( 'weights{ %d } must not contain small elements', index_object );
                    errorStruct.identifier = 'weighting:SmallWeights';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).weights = weights{ index_object }( : );

                % set independent properties
                objects( index_object ).weights_conj = conj( objects( index_object ).weights );

            end % for index_object = 1:numel( weights )

        end % function objects = weighting( weights )

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
                LTs = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) apply diagonal weighting matrix
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate diagonal weighting matrices
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % element-wise multiplication
                y{ index_object } = LTs( index_object ).weights .* x{ index_object };

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single diagonal weighting matrix
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
            % 2.) apply adjoint diagonal weighting matrix
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate diagonal weighting matrices
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % element-wise multiplication
                y{ index_object } = LTs( index_object ).weights_conj .* x{ index_object };

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single diagonal weighting matrix
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = adjoint_transform( LTs, x )

        %------------------------------------------------------------------
        % inverse transform (overload inverse_transform method)
        %------------------------------------------------------------------
        function y = inverse_transform( LTs, x )

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
                LTs = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) apply inverse diagonal weighting matrix
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate diagonal weighting matrices
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % element-wise division
                y{ index_object } = x{ index_object } ./ LTs( index_object ).weights;

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single diagonal weighting matrix
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = inverse_transform( LTs, x )

        %------------------------------------------------------------------
        % threshold
        %------------------------------------------------------------------
        function [ LTs, N_threshold ] = threshold( LTs, xis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
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

            end % for index_object = 1:numel( LTs )

        end % function LTs = threshold( LTs, xis )

    end % methods

end % classdef weighting < linear_transforms.invertible_linear_transform
