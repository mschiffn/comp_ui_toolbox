%
% superclass for all matrix-based discrete convolutions
%
% author: Martin F. Schiffner
% date: 2020-04-02
% modified: 2020-10-26
%
classdef matrix < linear_transforms.convolutions.convolution

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % dependent properties
        mat ( :, : )            % convolution matrix
        mat_adj ( :, : )        % adjoint convolution matrix

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = matrix( kernels, N_points, cut_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure valid number of input arguments
            narginchk( 2, 3 );

            % superclass ensures cell array for kernels
            % superclass ensures column vectors for kernels
            % superclass ensures nonempty positive integers for N_points

            % ensure existence of cut_off
            if nargin < 3
                cut_off = [];
            end

            % superclass ensures logical for cut_off

            %--------------------------------------------------------------
            % 2.) create matrix-based discrete convolutions
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.convolutions.convolution( kernels, N_points, cut_off );

            % iterate matrix-based discrete convolutions
            for index_object = 1:numel( objects )

                % set dependent properties
                objects( index_object ).mat = convmtx( objects( index_object ).kernel, objects( index_object ).N_points );
                if objects( index_object ).cut_off
                    objects( index_object ).mat = objects( index_object ).mat( ( objects( index_object ).M_kernel + 1 ):( end - objects( index_object ).M_kernel ), : );
                end
                objects( index_object ).mat_adj = objects( index_object ).mat';

            end % for index_object = 1:numel( objects )

        end % function objects = matrix( kernels, N_points, cut_off )

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
            % ensure class linear_transforms.convolutions.matrix (scalar)
            if ~( isa( LT, 'linear_transforms.convolutions.matrix' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.convolutions.matrix!';
                errorStruct.identifier = 'forward_transform_matrix:NoSingleMatrixConvolution';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward convolutions (single matrix)
            %--------------------------------------------------------------
            % apply forward transform using matrix
            y = LT.mat * x;

        end % function y = forward_transform_matrix( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single matrix)
        %------------------------------------------------------------------
        function y = adjoint_transform_matrix( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.convolutions.matrix (scalar)
            if ~( isa( LT, 'linear_transforms.convolutions.matrix' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.convolutions.matrix!';
                errorStruct.identifier = 'adjoint_transform_matrix:NoSingleMatrixConvolution';
                error( errorStruct );
            end

            % superclass ensures numeric matrix for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint convolutions (single matrix)
            %--------------------------------------------------------------
            % apply adjoint transform using matrix
            y = LT.mat_adj * x;

        end % function y = adjoint_transform_matrix( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef matrix < linear_transforms.convolutions.convolution
