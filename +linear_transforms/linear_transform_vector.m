%
% superclass for all linear transforms (vector processing)
%
% abstract superclass for all linear transforms
%
% author: Martin F. Schiffner
% date: 2020-01-29
% modified: 2020-04-16
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
