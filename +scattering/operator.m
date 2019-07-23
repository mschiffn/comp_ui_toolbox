%
% superclass for all scattering operators
%
% author: Martin F. Schiffner
% date: 2019-02-14
% modified: 2019-07-11
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
        incident_waves ( :, 1 ) syntheses.incident_wave             % incident waves
        E_M ( :, 1 ) physical_values.squarevolt                     % received energy

        % optional properties
        h_ref_aa ( 1, : ) discretizations.field                     % reference spatial transfer function w/ anti-aliasing filter (unique frequencies)
        E_M_aa ( :, 1 ) physical_values.squarevolt                  % received energy w/ anti-aliasing filter

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
                objects( index_object ).discretization = discretize( objects( index_object ).sequence, objects( index_object ).options.static.discretization );

                %----------------------------------------------------------
                % c) incident acoustic fields (unique frequencies)
                %----------------------------------------------------------
                objects( index_object ).incident_waves = syntheses.incident_wave( objects( index_object ).discretization );

                %----------------------------------------------------------
                % d) apply spatial anti-aliasing filter
                %----------------------------------------------------------
                if isa( objects( index_object ).discretization.spatial, 'discretizations.spatial_grid_symmetric' )
                    objects( index_object ).h_ref_aa = discretizations.anti_aliasing_filter( objects( index_object ).sequence.setup.xdc_array, objects( index_object ).sequence.setup.homogeneous_fluid, objects( index_object ).discretization.h_ref, objects( index_object ).options.momentary.anti_aliasing );
                end

                %----------------------------------------------------------
                % e) load or compute received energy
                %----------------------------------------------------------
                % create format string for filename
                str_format = sprintf( 'data/%s/spatial_%%s/E_M_spectral_%%s.mat', objects( index_object ).discretization.spatial.str_name );

                % load or compute received energy
                [ objects( index_object ).E_M, objects( index_object ).E_M_aa ] = auxiliary.compute_or_load_hash( str_format, @energy_rx, [ 2, 3 ], 1, objects( index_object ), objects( index_object ).discretization.spatial, objects( index_object ).discretization.spectral );

            end % for index_object = 1:numel( objects )

        end % function objects = operator( sequences, options )

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filters
        %------------------------------------------------------------------
        function spatiospectrals = anti_aliasing_filter( spatiospectrals, options_anti_aliasing )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.spatiospectral
            if ~isa( spatiospectrals, 'discretizations.spatiospectral' )
                errorStruct.message = 'spatiospectrals must be discretizations.spatiospectral!';
                errorStruct.identifier = 'anti_aliasing_filter:NoSpatiospectralDiscretizations';
                error( errorStruct );
            end

% TODO: symmetric spatial grids
            auxiliary.mustBeEqualSubclasses( 'discretizations.spatial_grid_symmetric', spatiospectrals.spatial )

            % ensure class scattering.options_anti_aliasing
            if ~isa( options_anti_aliasing, 'scattering.options_anti_aliasing' )
                errorStruct.message = 'options_anti_aliasing must be scattering.options_anti_aliasing!';
                errorStruct.identifier = 'discretize:NoAntiAliasingOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatiospectrals, options_anti_aliasing );

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filters
            %--------------------------------------------------------------
            % iterate spatiospectral discretizations
            for index_object = 1:numel( spatiospectrals )

                % apply spatial anti-aliasing filter via external function
                if options_anti_aliasing( index_object ).status == scattering.options_anti_aliasing_status.on
                    spatiospectrals( index_object ).h_ref_aa = discretizations.anti_aliasing_filter( spatiospectrals( index_object ).spatial, spatiospectrals( index_object ).h_ref, options_anti_aliasing( index_object ).parameter );
                end

            end % for index_object = 1:numel( spatiospectrals )

        end % function spatiospectrals = anti_aliasing_filter( spatiospectrals, options_anti_aliasing )

        %------------------------------------------------------------------
        % set properties of momentary scattering operator options
        %------------------------------------------------------------------
        function operators = set_properties_momentary( operators, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'set_properties_momentary:NoOperators';
                error( errorStruct );
            end

            % multiple operators / single varargin{ : }
            for index_arg = 1:numel( varargin )
                if ~isscalar( operators ) && isscalar( varargin{ index_arg } )
                    varargin{ index_arg } = repmat( varargin{ index_arg }, size( operators ) );
                end
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) set momentary scattering operator options
            %--------------------------------------------------------------
            % specify cell array for arguments
            args = cell( size( varargin ) );

            % iterate scattering operators
            for index_object = 1:numel( operators )

                % process arguments
                for index_arg = 1:numel( varargin )
                    args{ index_arg } = varargin{ index_arg }( index_object );
                end

                % set current momentary scattering options
                operators( index_object ).options = set_properties_momentary( operators( index_object ).options, args{ : } );

                % update data structures
                
            end

        end % function operators = set_properties_momentary( operators, options_momentary )

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
