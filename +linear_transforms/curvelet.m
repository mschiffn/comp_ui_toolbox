%
% compute two-dimensional discrete curvelet transform for various options
% author: Martin Schiffner
% date: 2016-08-13
%
classdef curvelet < linear_transforms.linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_lattice_axis
        N_scales
        N_angles_scale
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
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
            LT_curvelet@linear_transforms.linear_transform( N_coefficients, N_lattice, 'curvelet' );

            % internal properties
            LT_curvelet.N_lattice_axis	= N_lattice_axis;
            LT_curvelet.N_scales        = N_scales_temp;
            LT_curvelet.N_angles_scale	= N_angles_scale_temp;

        end

        %------------------------------------------------------------------
        % overload method: forward transform (forward DWAT)
        %------------------------------------------------------------------
        function y = forward_transform( LT_curvelet, x )

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

        end

        %------------------------------------------------------------------
        % overload method: adjoint transform (inverse DWAT)
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_curvelet, x )

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
        end

    end % methods

end % classdef curvelet < linear_transforms.linear_transform