%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-03-16
%
classdef operator_born < scattering.operator

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = operator_born( sequence, options )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            object@scattering.operator( sequence, options );

        end % function object = operator_born( sequence, options )

        %------------------------------------------------------------------
        % matrix multiplication (overload mtimes function)
        %------------------------------------------------------------------
        function u_M = mtimes( operator_born, fluctuations )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % TODO: compute rx signals for active elements for unique frequencies

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            % detect occupied grid points
            indices_occupied = find( abs( fluctuations ) > eps );

            u_M = cell( size( operator_born.discretization.spectral ) );

            % iterate sequential pulse-echo measurements
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing Born approximation of the recorded RF voltage signals (kappa)...', str_date_time );

            for index_measurement = 1:numel( operator_born.discretization.spectral )

                % initialize cell arrays
                u_M{ index_measurement } = cell( size( operator_born.discretization.spectral( index_measurement ).rx ) );

                %----------------------------------------------------------
                % 2.) spatial transfer function of the first array element
                %----------------------------------------------------------
                if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                    % compute complex-valued wavenumbers for unique frequencies
                    axis_f_unique = double( operator_born.discretization.spectral( index_measurement ).tx_unique.transfer_functions( 1 ).set_f.S );
                    axis_k_tilde_unique = operator_born.sequence.setup.absorption_model.compute_wavenumbers( axis_f_unique );

                    % spatial transfer function for unique frequencies and occupied grid points
                    h_rx_ref = syntheses.spatial_transfer_function( operator_born.discretization.spatial, axis_k_tilde_unique, 1, [], indices_occupied );

                end

                %----------------------------------------------------------
                % 3.) compute rx signals for active elements for unique frequencies
                %----------------------------------------------------------
                % check if mixes have identical frequency axes
                if isa( operator_born.options, '' )
                end

                % iterate received signals
%                 sets_f = [];
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    %------------------------------------------------------
                    % 1.) compute complex-valued wavenumbers
                    %------------------------------------------------------
                    sets_f( index_mix ) = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).transfer_functions( 1 ).set_f;
                    N_samples_f = abs( sets_f( index_mix ) );
                    axis_k_tilde = operator_born.sequence.setup.absorption_model.compute_wavenumbers( double( sets_f( index_mix ).S ) );
                    indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique{ index_mix };

                    %------------------------------------------------------
                    % 2.) 
                    %------------------------------------------------------
                    % compute signals received by each active element in the mix
                    delta_A = operator_born.discretization.spatial.grids_elements( 1 ).delta_V;
                    delta_V = operator_born.discretization.spatial.grid_FOV.delta_V;

                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );
                    u_act = zeros( N_elements_active, N_samples_f );

                    factor_interp_tx = round( operator_born.sequence.setup.xdc_array.element_pitch_axis ./ operator_born.discretization.spatial.grid_FOV.delta_axis(1:(end-1)) );
                    for index_active = 1:N_elements_active

                        % index of array element
                        index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % receiver electromechanical transfer function
                        transfer_function_rx = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).transfer_functions( index_active );

                        %--------------------------------------------------
                        % spatial transfer function of the active array element
                        %--------------------------------------------------
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric grid
                            %----------------------------------------------
                            % shift in grid points required for current array element
                            delta_lattice_points = ( index_element - 1 ) * factor_interp_tx;

                            % compute summand for the incident pressure field
                            index_start = operator_born.discretization.spatial.grid_FOV.N_points_axis(1) - ( operator_born.sequence.setup.xdc_array.N_elements - 1 ) * factor_interp_tx + 1;
                            index_stop = index_start + delta_lattice_points - 1;
                            h_rx = [ h_rx_ref( :, index_stop:-1:index_start, indices_f_to_unique ), h_rx_ref( :, 1:(end - delta_lattice_points), indices_f_to_unique ) ];

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % spatial impulse response of the active array element
                            h_rx = syntheses.spatial_transfer_function( operator_born.discretization.spatial, axis_k_tilde, index_element, [], indices_occupied );

                        end

                        % avoid spatial aliasing
%                         indicator_aliasing = flag > real( axis_k_tilde( index_f ) );
%                         indicator_aliasing = indicator_aliasing .* ( 1 - ( real( axis_k_tilde( index_f ) ) ./ flag).^2 );

                        for index_f = 1:N_samples_f
                            Phi_act = 2 * delta_A * delta_V * axis_k_tilde( index_f ).^2 .* transfer_function_rx.samples( index_f ) .* h_rx( :, :, index_f ) .* operator_born.p_incident( index_measurement ).values{ indices_f_to_unique( index_f ) };
                            u_act( index_active, index_f ) = Phi_act( : ).' * fluctuations;
                        end

                    end % for index_active = 1:N_elements_active

                    % compute mixed RF voltage signal
                    u_M{ index_measurement }{ index_mix } = sum( u_act, 1 );

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                %----------------------------------------------------------
                % 3.) create truncated Fourier series
                %----------------------------------------------------------
                u_M{ index_measurement } = physical_values.fourier_series_truncated( sets_f, u_M{ index_measurement } );

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

            % remove cell array for single entry
            if numel( u_M ) == 1
                u_M = u_M{ 1 };
            end

        end % function u_M = mtimes( operator_born, fluctuations )

        %------------------------------------------------------------------
        % received energy
        %------------------------------------------------------------------
        function energy_rx()
        end

        %------------------------------------------------------------------
        % point spread function (PSF)
        %------------------------------------------------------------------
        function out = psf( operator_born, indices )
        end

	end % methods

end % classdef operator_born < scattering.operator
