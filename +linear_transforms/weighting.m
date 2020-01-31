%
% compute effect of diagonal weighting matrix
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-01-30
%
classdef weighting < linear_transforms.linear_transform_matrix

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

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for weights
            if ~iscell( weights )
                weights = mat2cell( weights, size( weights, 1 ), ones( 1, size( weights, 2 ) ) );
            end

            %--------------------------------------------------------------
            % 2.) create diagonal weighting matrices
            %--------------------------------------------------------------
            % numbers of weights
            N_weights = cellfun( @numel, weights );

            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_weights, N_weights );

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

                % set dependent properties
                objects( index_object ).weights_conj = conj( objects( index_object ).weights );

            end % for index_object = 1:numel( weights )

        end % function objects = weighting( weights )

        %------------------------------------------------------------------
        % normalization
        %------------------------------------------------------------------
        function LTs = normalize( LTs, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.weighting
            if ~isa( LTs, 'linear_transforms.weighting' )
                errorStruct.message = 'LTs must be linear_transforms.weighting!';
                errorStruct.identifier = 'normalize:NoWeightings';
                error( errorStruct );
            end

            % ensure class regularization.options.normalization
            if ~isa( options, 'regularization.options.normalization' )
                errorStruct.message = 'options must be regularization.options.normalization!';
                errorStruct.identifier = 'normalize:NoNormalizationOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, options );

            %--------------------------------------------------------------
            % 2.) apply normalization
            %--------------------------------------------------------------
            % iterate linear transforms
            for index_object = 1:numel( LTs )

                % check type of normalization
                if isa( options( index_object ), 'regularization.options.normalization_off' )

                    %------------------------------------------------------
                    % a) no normalization
                    %------------------------------------------------------
                    % do not modify linear transform

                elseif isa( options( index_object ), 'regularization.options.normalization_threshold' )

                    %------------------------------------------------------
                    % b) apply threshold to inverse weighting matrix
                    %------------------------------------------------------
                    [ LTs( index_object ), N_threshold ] = threshold( LTs( index_object ), options( index_object ).threshold );

                else

                    %------------------------------------------------------
                    % c) unknown normalization settings
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Class of options( %d ) is unknown!', index_object );
                    errorStruct.identifier = 'normalize:UnknownOptionsClass';
                    error( errorStruct );

                end % if isa( options( index_object ), 'regularization.options.normalization_off' )

            end % for index_object = 1:numel( LTs )

        end % function LTs = normalize( LTs, options )

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
            % ensure class linear_transforms.weighting (scalar)
            if ~( isa( LT, 'linear_transforms.weighting' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.weighting!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleDiagonalConcatenation';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward diagonal weighting (single matrix)
            %--------------------------------------------------------------
            % element-wise multiplication
            y = LT.weights .* x;

        end % function y = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function y = adjoint_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.weighting (scalar)
            if ~( isa( LT, 'linear_transforms.weighting' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.weighting!';
                errorStruct.identifier = 'adjoint_transform_matrix:NoSingleComposition';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint diagonal weighting (single matrix)
            %--------------------------------------------------------------
            % element-wise multiplication
            y = LT.weights_conj .* x;

        end % function y = adjoint_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % inverse transform (single matrix)
        %------------------------------------------------------------------
        function y = inverse_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.weighting (scalar)
            if ~( isa( LT, 'linear_transforms.weighting' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.weighting!';
                errorStruct.identifier = 'inverse_transform_matrix:NoSingleComposition';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint diagonal weighting (single matrix)
            %--------------------------------------------------------------
            % element-wise division
            y = x ./ LT.weights;

        end % function y = inverse_transform_matrix( LT, x )

	end % methods (Access = protected, Hidden)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

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

        end % function [ LTs, N_threshold ] = threshold( LTs, xis )

	end % methods (Access = private, Hidden)

end % classdef weighting < linear_transforms.linear_transform_matrix
