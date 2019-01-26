%
% compute two-dimensional discrete wave atom transform for various options
% author: Martin Schiffner
% date: 2016-08-13
%
classdef wave_atom < linear_transforms.linear_transform
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_lattice_axis
        transform_type
        N_layers
    end % properties
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_wave_atom = wave_atom( N_lattice_axis, transform_type )

            % total number of lattice points
            N_lattice = N_lattice_axis(1) * N_lattice_axis(2);

            % number of layers
            if strcmp( transform_type, 'ortho' )
                % orthobasis
                N_layers_temp = 1;
            elseif strcmp( transform_type, 'directional' )
                % real-valued frame with single oscillation direction
                N_layers_temp = 2;
            elseif strcmp( transform_type, 'complex' )
                % complex-valued frame
                N_layers_temp = 4;
            end

            % number of transform coefficients
            N_coefficients = N_layers_temp * N_lattice;

            % create name string
            str_name = sprintf('%s_%s', 'wave_atom', transform_type);

            % constructor of superclass
            LT_wave_atom@linear_transforms.linear_transform( N_coefficients, N_lattice, str_name );

            % internal properties
            LT_wave_atom.N_lattice_axis = N_lattice_axis;
            LT_wave_atom.transform_type = transform_type;
            LT_wave_atom.N_layers       = N_layers_temp;

        end

        %------------------------------------------------------------------
        % overload method: forward transform (forward DWAT)
        %------------------------------------------------------------------
        function y = forward_transform( LT_wave_atom, x )

            x = reshape( x, [LT_wave_atom.N_lattice_axis(2), LT_wave_atom.N_lattice_axis(1)] );
            y = fwa2sym( x, 'q', LT_wave_atom.transform_type );
        end

        %------------------------------------------------------------------
        % overload method: adjoint transform (inverse DWAT)
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_wave_atom, x )

            x = reshape( x, [LT_wave_atom.N_lattice_axis(2), LT_wave_atom.N_lattice_axis(1), LT_wave_atom.N_layers] );
            y = iwa2sym( x, 'q', LT_wave_atom.transform_type );
        end

    end % methods

end % classdef wave_atom