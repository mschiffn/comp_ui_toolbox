%
% compute two-dimensional discrete curvelet transform for various options
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-01-31
%
classdef curvelet < linear_transforms.linear_transform_vector

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_lattice_axis
        N_scales
        N_angles_scale

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_curvelet = curvelet( N_lattice_axis )

            % total number of lattice points
            N_lattice = N_lattice_axis(1) * N_lattice_axis(2);

            index_scale_max = ceil( log2( min( N_lattice_axis ) ) );	% maximum scale index (fines scale)
            N_scales_temp = index_scale_max - 3;                             % number of scales to be investigated (including coarsest scale)
            N_angles_2nd_coarse = 16;                                   % number of angles at the 2nd coarsest level

            N_angles_scale_temp = [1, N_angles_2nd_coarse * 2.^(ceil((0:(N_scales_temp-2))/2))];
            N_angles_scale_temp(N_scales_temp) = 1;

            % dummy forward curvelet transform
            C = fdct_wrapping( zeros(N_lattice_axis(2), N_lattice_axis(1)) );

            % coarsest scale
            y = C{1,1}{1,1}(:);

            % intermediate scales
            for index_scale = 2:(N_scales_temp - 1)

                for index_angle = 1:N_angles_scale_temp(index_scale)

                    y = [y; C{1, index_scale}{1, index_angle}(:)];
                end
            end

            % finest scale
            y = [y; C{1, N_scales_temp}{1, 1}(:)];

            % number of transform coefficients
            N_coefficients = size(y, 1);

            % constructor of superclass
            LT_curvelet@linear_transforms.linear_transform_vector( N_coefficients, N_lattice );

            % internal properties
            LT_curvelet.N_lattice_axis	= N_lattice_axis;
            LT_curvelet.N_scales        = N_scales_temp;
            LT_curvelet.N_angles_scale	= N_angles_scale_temp;

        end

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single vector)
        %------------------------------------------------------------------
        function y = forward_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.fourier (scalar)
            if ~( isa( LT, 'linear_transforms.fourier' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.fourier!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleFourierTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward curvelet transform (single vector)
            %--------------------------------------------------------------
            x = reshape( x, [LT_curvelet.N_lattice_axis(2), LT_curvelet.N_lattice_axis(1)] );
            C = fdct_wrapping( x );

            % coarsest scale
            y = C{1,1}{1,1}(:);

            % intermediate scales
            for index_scale = 2:(LT_curvelet.N_scales - 1)
                for index_angle = 1:LT_curvelet.N_angles_scale(index_scale)

                    y = [y; C{1, index_scale}{1, index_angle}(:)];
                end
            end
            
            % finest scale
            y = [y; C{1, LT_curvelet.N_scales}{1, 1}(:)];

        end % function y = forward_transform_vector( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single vector)
        %------------------------------------------------------------------
        function y = adjoint_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.fourier (scalar)
            if ~( isa( LT, 'linear_transforms.fourier' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.fourier!';
                errorStruct.identifier = 'adjoint_transform_vector:NoSingleFourierTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint curvelet transform (single vector)
            %--------------------------------------------------------------
            % create dummy data structure to check number of elements
            C = fdct_wrapping(zeros(LT_curvelet.N_lattice_axis(2), LT_curvelet.N_lattice_axis(1)), 0, 2);

            % coarsest scale
            N_elements_act = numel(C{1,1}{1,1});
            C{1,1}{1,1}(:) = x(1:N_elements_act);
            index_element_act = N_elements_act + 1;

            % intermediate scales
            for index_scale = 2:(LT_curvelet.N_scales - 1)
                for index_angle = 1:LT_curvelet.N_angles_scale(index_scale)
                    
                    N_elements_act = size(C{1, index_scale}{1, index_angle});
                    index_start = index_element_act;
                    index_stop = index_start + N_elements_act(1) * N_elements_act(2) - 1;

                    C{1, index_scale}{1, index_angle} = zeros(N_elements_act);
                    C{1, index_scale}{1, index_angle}(:) = x(index_start:index_stop);
                    index_element_act = index_stop + 1;
                end
            end

            % finest scale
            C{1, LT_curvelet.N_scales}{1,1}(:) = x(index_element_act:end);

            y = ifdct_wrapping(C, 0, LT_curvelet.N_lattice_axis(2), LT_curvelet.N_lattice_axis(1));
            y = y(:);

        end % function y = adjoint_transform_vector( LT, x )

	end % methods (Access = protected, Hidden)

end % classdef curvelet < linear_transforms.linear_transform