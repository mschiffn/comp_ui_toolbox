%
% superclass for all scattering operators
%
% author: Martin F. Schiffner
% date: 2019-02-14
% modified: 2019-05-27
%
classdef operator

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% public properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = public)

        % independent properties
        options ( 1, 1 ) scattering.options                         % scattering operator options

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% private properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sequence %( 1, 1 ) pulse_echo_measurements.sequence         % pulse-echo measurement sequence

        % dependent properties
        discretization ( 1, 1 ) discretizations.spatiospectral      % results of the spatiospectral discretization
        incident_waves ( :, : ) syntheses.incident_wave             % incident waves
        E_M ( :, 1 ) physical_values.squarevolt                     % received energy

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = operator( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.sequence
            if ~isa( sequences, 'pulse_echo_measurements.sequence' )
                errorStruct.message = 'sequences must be pulse_echo_measurements.sequence!';
                errorStruct.identifier = 'operator:NoSequences';
                error( errorStruct );
            end

            % ensure class scattering.options
            if ~isa( options, 'scattering.options' )
                errorStruct.message = 'options must be scattering.options!';
                errorStruct.identifier = 'operator:NoOptions';
                error( errorStruct );
            end

            % multiple sequences / single options
            if ~isscalar( sequences ) && isscalar( options )
                options = repmat( options, size( sequences ) );
            end

            % single sequences / multiple options
            if isscalar( sequences ) && ~isscalar( options )
                sequences = repmat( sequences, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) create scattering operators
            %--------------------------------------------------------------
            % repeat default scattering operator
            objects = repmat( objects, size( sequences ) );

            % iterate scattering operators
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).sequence = sequences( index_object );
                objects( index_object ).options = options( index_object );

                %----------------------------------------------------------
                % b) spatiospectral discretization of the sequence
                %----------------------------------------------------------
                objects( index_object ).discretization = discretize( objects( index_object ).sequence, objects( index_object ).options.discretization );

                %----------------------------------------------------------
                % c) incident acoustic fields (unique frequencies)
                %----------------------------------------------------------
                objects( index_object ).incident_waves = syntheses.incident_wave( objects( index_object ).discretization );

                %----------------------------------------------------------
                % d) load or compute received energy
                %----------------------------------------------------------
                % create format string for filename
                str_format = sprintf( 'data/%s/spatial_%%s/E_M_spectral_%%s.mat', objects( index_object ).discretization.spatial.str_name );

                % load or compute received energy
                objects( index_object ).E_M = auxiliary.compute_or_load_hash( str_format, @energy_rx, [ 2, 3 ], 1, objects( index_object ), objects( index_object ).discretization.spatial, objects( index_object ).discretization.spectral );

            end % for index_object = 1:numel( objects )

        end % function objects = operator( sequences, options )

        %------------------------------------------------------------------
        % transform point spread function (TPSF)
        %------------------------------------------------------------------
        function [ theta_hat, E_M, adjointness ] = tpsf( operators, indices, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, indices );

            %--------------------------------------------------------------
            % 2.) compute TPSFs
            %--------------------------------------------------------------
            % specify cell array for psf
            theta_hat = cell( size( operators ) );
            E_M = cell( size( operators ) );
            adjointness = cell( size( operators ) );

            % iterate scattering operators
            for index_object = 1:numel( operators )

                % number of PSFs
                N_psf = numel( indices{ index_object } );

                % initialize coefficient vectors and output with zeros
                theta_hat{ index_object } = zeros( operators( index_object ).discretization.spatial.grid_FOV.N_points, N_psf );
%                 if options.material_parameter ~= 0
%                     theta = zeros(N_lattice, N_tpsf);
%                     theta_recon = zeros(N_lattice, N_tpsf);
%                     gamma_recon = zeros(N_lattice, N_tpsf);
%                 else
%                     theta = zeros(2*N_lattice, N_tpsf);
%                     theta_recon = zeros(2*N_lattice, N_tpsf);
%                     gamma_recon = zeros(2*N_lattice, N_tpsf);
%                 end
                gamma_kappa = zeros( operators( index_object ).discretization.spatial.grid_FOV.N_points, 1 );

                E_M{ index_object } = zeros( 1, N_psf );
                adjointness{ index_object } = zeros( 1, N_psf );

                % iterate grid points
                for index_grid = 1:N_psf

                    % specify current gamma_kappa
                    gamma_kappa( indices{ index_object }( index_grid ) ) = 1;

                    % compute forward scattering
                    u_M = forward( operators( index_object ), gamma_kappa );
                    E_M{ index_object }( index_grid ) = energy( u_M );

                    % compute adjoint scattering
                    theta_hat{ index_object }( :, index_grid ) = adjoint( operators( index_object ), u_M );
                    adjointness{ index_object }( index_grid ) = E_M{ index_object }( index_grid ) - theta_hat{ index_object }( indices{ index_object }( index_grid ), index_grid );

                    % delete current gamma_kappa
                    gamma_kappa( indices{ index_object }( index_grid ) ) = 0;

                    figure( index_grid );
                    imagesc( squeeze( reshape( double( abs( theta_hat{ index_object }( :, index_grid ) ) ), operators( index_object ).discretization.spatial.grid_FOV.N_points_axis ) ) );

                end % for index_grid = 1:N_psf

            end % for index_object = 1:numel( operators )

            % reshape results
%         if options.material_parameter ~= 0
%             gamma_recon = reshape(gamma_recon, [N_lattice_axis(2), N_lattice_axis(1), N_tpsf]);
%             theta_recon = reshape(theta_recon, [N_lattice_axis(2), N_lattice_axis(1), N_tpsf]);
%         else
%             gamma_recon = reshape(gamma_recon, [N_lattice_axis(2), 2*N_lattice_axis(1), N_tpsf]);
%             theta_recon = reshape(theta_recon, [N_lattice_axis(2), 2*N_lattice_axis(1), N_tpsf]);
%         end

            % avoid cell array for single operators
            if isscalar( operators )
                theta_hat = theta_hat{ 1 };
                E_M = E_M{ 1 };
                adjointness = adjointness{ 1 };
            end

        end % function [ theta_hat, E_M, adjointness ] = tpsf( operators, indices, varargin )

        %------------------------------------------------------------------
        % received energy
        %------------------------------------------------------------------
        function E_M = energy_rx( operators )

            % extract number of grid points
            N_points = operators.discretization.spatial.grid_FOV.N_points;

            % initialize relative spatial fluctuations with zeros
            gamma_kappa = zeros( N_points, 1 );
            E_M = zeros( N_points, 1 );

            % iterate grid points
            for index_point = 1:N_points

                % specify fluctuation
                gamma_kappa( index_point ) = 1;

                % compute forward scattering
                u_M = forward( operators, gamma_kappa );

                % compute received energy
                E_M( index_point ) = energy( u_M );

                % reset fluctuation
                gamma_kappa( index_point ) = 0;
% TODO: save results regularly
            end % for index_point = 1:N_points

        end % function E_M = energy_rx( operators )

    end % methods

end % classdef operator
