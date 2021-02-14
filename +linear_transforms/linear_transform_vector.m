%
% superclass for all linear transforms (vector processing)
%
% abstract superclass for all linear transforms
%
% author: Martin F. Schiffner
% date: 2020-01-29
% modified: 2020-11-05
%
classdef (Abstract) linear_transform_vector < linear_transforms.linear_transform_matrix

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = linear_transform_vector( N_coefficients, N_points )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures nonempty positive integers for N_coefficients
            % superclass ensures nonempty positive integers for N_points
            % superclass ensures equal number of dimensions and sizes for N_coefficients and N_points

            %--------------------------------------------------------------
            % 2.) create linear transforms (matrix processing)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.linear_transform_matrix( N_coefficients, N_points )

        end % function objects = linear_transform_vector( N_coefficients, N_points )

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
            % calling function ensures class linear_transforms.linear_transform_matrix (scalar) for LT
            % calling function ensures numeric matrix for x
            % calling function ensures equal numbers of points

            % ensure class linear_transforms.linear_transform_vector (scalar)
            if ~( isa( LT, 'linear_transforms.linear_transform_vector' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.linear_transform_vector!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleVectorTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute forward transform (single matrix)
            %--------------------------------------------------------------
            % number of vectors to transform
            N_vectors = size( x, 2 );

            % initialize results with zeros
            y = zeros( LT.N_coefficients, N_vectors );

            % iterate vectors
            for index_vector = 1:N_vectors

                % call forward transform for single vector
                y( :, index_vector ) = forward_transform_vector( LT, x( :, index_vector ) );

            end % for index_vector = 1:N_vectors

        end % function y = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function y = adjoint_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.linear_transform_matrix (scalar) for LT
            % calling function ensures numeric matrix for x
            % calling function ensures equal numbers of coefficients

            % ensure class linear_transforms.linear_transform_vector (scalar)
            if ~( isa( LT, 'linear_transforms.linear_transform_vector' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.linear_transform_vector!';
                errorStruct.identifier = 'adjoint_transform_matrix:NoSingleVectorTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute adjoint transform (single matrix)
            %--------------------------------------------------------------
            % number of vectors to transform
            N_vectors = size( x, 2 );

            % initialize results with zeros
            y = zeros( LT.N_points, N_vectors );

            % iterate vectors
            for index_vector = 1:N_vectors

                % call adjoint transform for single vector
                y( :, index_vector ) = adjoint_transform_vector( LT, x( :, index_vector ) );

            end % for index_vector = 1:N_vectors

        end % function y = adjoint_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single matrix)
        %------------------------------------------------------------------
        function display_coefficients_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.linear_transform_matrix (scalar) for LT
            % calling function ensures numeric matrix for x
            % calling function ensures equal numbers of coefficients

            % ensure class linear_transforms.linear_transform_vector (scalar)
            if ~( isa( LT, 'linear_transforms.linear_transform_vector' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.linear_transform_vector!';
                errorStruct.identifier = 'display_coefficients_matrix:NoSingleVectorTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display coefficients (single matrix)
            %--------------------------------------------------------------
            % number of vectors to transform
            N_vectors = size( x, 2 );

            % iterate vectors
            for index_vector = 1:N_vectors

                % call display coefficients for single vector
                display_coefficients_vector( LT, x( :, index_vector ) );

            end % for index_vector = 1:N_vectors

        end % function display_coefficients_matrix( LT, x )

        %------------------------------------------------------------------
        % relative RMSEs of s largest expansion coefficients (single matrix)
        %------------------------------------------------------------------
        function [ rel_RMSEs, axis_s ] = rel_RMSE_matrix( LT, x, y, N_points_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.linear_transform_matrix (scalar) for LT
            % calling function ensures numeric matrix for x
            % calling function ensures equal numbers of points in x
            % calling function ensures valid number of evaluation points N_points_s

            %--------------------------------------------------------------
            % 2.) compute relative RMSEs of s largest expansion coefficients (single matrix)
            %--------------------------------------------------------------
            % number of vectors
            N_vectors = size( x, 2 );

            % create current sparsity axis
            axis_s = round( logspace( 0, log10( LT.N_coefficients ), N_points_s ) );

            % initialize results w/ zeros
            rel_RMSEs = zeros( N_points_s, N_vectors );

            % iterate vectors
            for index_vector = 1:N_vectors

                % call forward transform for single vector
                rel_RMSEs( :, index_vector ) = rel_RMSE_vector( LT, x( :, index_vector ), y( :, index_vector ), axis_s );

            end % for index_vector = 1:N_vectors

        end % function [ rel_RMSEs, axis_s ] = rel_RMSE_matrix( LT, x, y, N_points_s )

        %------------------------------------------------------------------
        % relative RMSEs of s largest expansion coefficients (single vector)
        %------------------------------------------------------------------
        function rel_RMSEs = rel_RMSE_vector( LT, x, y, axis_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures

            %--------------------------------------------------------------
            % 2.) compute relative RMSEs of s largest expansion coefficients (single vector)
            %--------------------------------------------------------------
            % energy of samples
            x_energy = norm( x );

            % sort absolute values of transform coefficients (descending order)
            [ ~, indices_sorted ] = sort( abs( y ), 1, 'descend' );

            % allocate memory
            y_act = zeros( LT.N_coefficients, 1 );
            rel_RMSEs = zeros( 1, numel( axis_s ) );

            % create sparse coefficient vector, compute approx. image
            for index_s_act = 1:numel( axis_s )

                % copy desired transform coefficients
                indices_act = indices_sorted( 1:axis_s( index_s_act ) );
                y_act( indices_act ) = y( indices_act );

                % apply adjoint transform and compute approximated image
                x_act = adjoint_transform_vector( LT, y_act );

                % compute relative RMSE
                rel_RMSEs( index_s_act ) = norm( x - x_act ) / x_energy;

            end % for index_s_act = 1:numel( axis_s )

        end % function rel_RMSEs = rel_RMSE_vector( LT, x, y, axis_s )

    end % methods (Access = protected, Hidden)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single vector)
        %------------------------------------------------------------------
        y = forward_transform_vector( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single vector)
        %------------------------------------------------------------------
        y = adjoint_transform_vector( LT, x )

        %------------------------------------------------------------------
        % display coefficients (single vector)
        %------------------------------------------------------------------
        display_coefficients_vector( LT, x )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) linear_transform_matrix < linear_transforms.linear_transform
