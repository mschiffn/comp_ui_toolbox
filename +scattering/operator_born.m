%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-04-09
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
%             indices_occupied = find( abs( fluctuations ) > eps );

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing Born approximation of the recorded RF voltage signals (kappa)...', str_date_time );

            % geometric volume elements
            if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )
                delta_A = operator_born.discretization.spatial.grids_elements( 1 ).cell_ref.volume;
                delta_V = operator_born.discretization.spatial.grid_FOV.cell_ref.volume;
            end

            % specify cell array for u_M
            u_M = cell( size( operator_born.discretization.spectral ) );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                % extract unique frequencies
                axis_f_unique = operator_born.discretization.spectral( index_measurement ).tx_unique.excitation_voltages.axis;

                %----------------------------------------------------------
                % 1.) compute rx signals for active elements for unique frequencies
                %----------------------------------------------------------
                % check if mixes have identical frequency axes
%                 if isa( operator_born.options, '' )
%                 end

                %----------------------------------------------------------
                % 2.) compute complex-valued wavenumbers
                %----------------------------------------------------------
                impulse_responses_rx = reshape( [ operator_born.discretization.spectral( index_measurement ).rx.impulse_responses ], size( operator_born.discretization.spectral( index_measurement ).rx ) );
                axes_f = reshape( [ impulse_responses_rx.axis ], size( operator_born.discretization.spectral( index_measurement ).rx ) );
                N_samples_f = abs( axes_f );
                axes_k_tilde = compute_wavenumbers( operator_born.sequence.setup.absorption_model, axes_f );

                % specify cell arrays for u_M
                u_M{ index_measurement } = cell( size( operator_born.discretization.spectral( index_measurement ).rx ) );

                % iterate received mixes
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    %------------------------------------------------------
                    % 1.) map frequencies to unique frequencies
                    %------------------------------------------------------
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );
                    indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique{ index_mix };
                    pressure_act = subsample( operator_born.incident_waves( index_measurement ).p_incident, indices_f_to_unique );
                    prefactors = discretizations.signal_matrix( axes_f( index_mix ), repmat( 2 * delta_A * delta_V * axes_k_tilde( index_mix ).members.^2, [ N_elements_active, 1 ] ) );
                    prefactors = prefactors .* impulse_responses_rx( index_mix );

                    %------------------------------------------------------
                    % 2.) 
                    %------------------------------------------------------
                    u_act = zeros( N_elements_active, N_samples_f( index_mix ) );

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % index of active array element
                        index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        %--------------------------------------------------
                        % spatial transfer function of the active array element
                        %--------------------------------------------------
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift in grid points required for current array element
                            index_element_axis = inverse_index_transform( operator_born.sequence.setup.xdc_array, index_element - 1 );
                            N_points_shift_axis = index_element_axis .* operator_born.discretization.spatial.N_points_per_pitch_axis;

                            % shift reference spatial transfer function to infer that of the active array element
                            h_rx = shift( operator_born.incident_waves( index_measurement ).h_ref, operator_born.discretization.spatial, N_points_shift_axis );

                            % get current frequencies
                            h_rx = subsample( h_rx, indices_f_to_unique );

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % spatial impulse response of the active array element
                            h_rx_test = discretizations.spatial_transfer_function( operator_born.discretization.spatial.grids_elements( index_element ), operator_born.discretization.spatial.grid_FOV, operator_born.sequence.setup.absorption_model, axis_f_unique );

                        end

                        % avoid spatial aliasing
                        if operator_born.options.spatial_aliasing == scattering.options_aliasing.exclude
                            indicator_aliasing = flag > real( axes_k_tilde( index_f ) );
                            indicator_aliasing = indicator_aliasing .* ( 1 - ( real( axes_k_tilde( index_f ) ) ./ flag).^2 );
                        end

                        % compute matrix-vector product
                        temp = h_rx .* pressure_act;
                        Phi_act = reshape( reshape( prefactors.samples, [ ones( 1, operator_born.discretization.spatial.grid_FOV.N_dimensions ), N_samples_f( index_mix ) ] ) .* temp.samples, [ operator_born.discretization.spatial.grid_FOV.N_points, N_samples_f( index_mix ) ] );
                        u_act( index_active, : ) = Phi_act.' * fluctuations;

                    end % for index_active = 1:N_elements_active

                    % compute mixed RF voltage signal
                    u_M{ index_measurement }{ index_mix } = sum( u_act, 1 );

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                %----------------------------------------------------------
                % 3.) create signal matrices
                %----------------------------------------------------------
                u_M{ index_measurement } = discretizations.signal_matrix( axes_f, u_M{ index_measurement } );

                try
                    u_M{ index_measurement } = merge( 1, u_M{ index_measurement } );
                catch
                end

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

            % avoid cell array for single pulse-echo measurement
            if isscalar( u_M )
                u_M = u_M{ 1 };
            end

        end % function u_M = mtimes( operator_born, fluctuations )

        %------------------------------------------------------------------
        % adjoint multiplication (overload mtimes function)
        %------------------------------------------------------------------
        function fluctuations = adjoint( operator_born, u_M )


        end % function fluctuations = adjoint( operator_born, u_M )

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
