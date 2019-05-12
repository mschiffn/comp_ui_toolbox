%
% superclass for all scattering operators
%
% author: Martin F. Schiffner
% date: 2019-02-14
% modified: 2019-05-11
%
classdef operator

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sequence %( 1, 1 ) pulse_echo_measurements.sequence         % pulse-echo measurement sequence
        options ( 1, 1 ) scattering.options                         % scattering operator options

        % dependent properties
        discretization ( 1, 1 ) discretizations.spatiospectral      % results of the spatiospectral discretization
        incident_waves ( :, : ) syntheses.incident_wave             % incident waves
        E_rx ( :, 1 ) %physical_values.volt

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = operator( sequence, options )
% TODO: vectorize
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.sequence (scalar)
            if ~( isa( sequence, 'pulse_echo_measurements.sequence' ) && isscalar( sequence ) )
                errorStruct.message     = 'sequence must be a single pulse_echo_measurements.sequence!';
                errorStruct.identifier	= 'operator:NoScalarSequence';
                error( errorStruct );
            end

            % ensure class scattering.options (scalar)
            if ~( isa( options, 'scattering.options' ) && isscalar( options ) )
                errorStruct.message     = 'options must be a single scattering.options!';
                errorStruct.identifier	= 'operator:NoScalarOptions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            object.sequence = sequence;
            object.options = options;

            % TODO: check for identical recording time intervals / identical frequency intervals
            % TODO: check method to determine Fourier coefficients
            % TODO: find active elements and compute impulse responses
            % check for identical frequency axes identical?
            % TODO: check for valid spatial discretization (sampling theorem)
            %--------------------------------------------------------------
            % 3.) spatiospectral discretization of the sequence
            %--------------------------------------------------------------
            object.discretization = discretize( object.sequence, object.options.discretization );

            %--------------------------------------------------------------
            % 4.) incident acoustic fields (unique frequencies)
            %--------------------------------------------------------------
            object.incident_waves = syntheses.incident_wave( object.sequence.setup, object.discretization );

            %--------------------------------------------------------------
            % 5.) compute received energy
            %--------------------------------------------------------------
            object.E_rx = energy_rx( object );

        end % function object = operator( sequence, options )

        %------------------------------------------------------------------
        % point spread function (PSF)
        %------------------------------------------------------------------
        function [ out, E_rx, adjointness ] = psf( operators, indices )

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
            % 2.) compute PSFs
            %--------------------------------------------------------------
            % specify cell array for psf
            out = cell( size( operators ) );
            E_rx = cell( size( operators ) );
            adjointness = cell( size( operators ) );

            % iterate scattering operators
            for index_object = 1:numel( operators )

                % number of PSFs
                N_psf = numel( indices{ index_object } );

                % initialize coefficient vectors and output with zeros
                out{ index_object } = zeros( operators( index_object ).discretization.spatial.grid_FOV.N_points, N_psf );
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

                E_rx{ index_object } = zeros( 1, N_psf );
                adjointness{ index_object } = zeros( 1, N_psf );

                % iterate grid points
                for index_grid = 1:N_psf

                    % specify current gamma_kappa
                    gamma_kappa( indices{ index_object }( index_grid ) ) = 1;

                    % compute forward scattering
                    u_M = forward( operators( index_object ), gamma_kappa );
                    E_rx{ index_object }( index_grid ) = energy( u_M );

                    % compute adjoint scattering
                    out{ index_object }( :, index_grid ) = adjoint( operators( index_object ), u_M );
                    adjointness{ index_object }( index_grid ) = E_rx{ index_object }( index_grid ) - out{ index_object }( indices{ index_object }( index_grid ), index_grid );

                    % delete current gamma_kappa
                    gamma_kappa( indices{ index_object }( index_grid ) ) = 0;

                    figure( index_grid );
                    imagesc( squeeze( reshape( double( abs( out{ index_object }( :, index_grid ) ) ), operators( index_object ).discretization.spatial.grid_FOV.N_points_axis ) ) );

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
                out = out{ 1 };
                E_rx = E_rx{ 1 };
                adjointness = adjointness{ 1 };
            end

        end % function [ out, E_rx, adjointness ] = psf( operators, indices )

        %------------------------------------------------------------------
        % received energy
        %------------------------------------------------------------------
        function E_rx = energy_rx( operators )

            % extract number of grid points
            N_points = operators.discretization.spatial.grid_FOV.N_points;

            % initialize relative spatial fluctuations with zeros
            gamma_kappa = zeros( N_points, 1 );
            E_rx = zeros( N_points, 1 );

            % iterate grid points
            for index_point = 1:N_points

                % specify fluctuation
                gamma_kappa( index_point ) = 1;

                % compute forward scattering
                u_M = forward( operators, gamma_kappa );

                % compute received energy
                E_rx( index_point ) = energy( u_M );

                % reset fluctuation
                gamma_kappa( index_point ) = 0;
% TODO: save results regularly
            end % for index_point = 1:N_points

        end % function E_rx = energy_rx( operators )

    end % methods

end % classdef operator
