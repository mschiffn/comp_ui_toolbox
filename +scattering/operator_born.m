%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-05-05
%
classdef operator_born < scattering.operator

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
        % forward scattering (overload forward function)
        %------------------------------------------------------------------
        function u_M = forward( operator_born, fluctuations )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if ~isnumeric( fluctuations )
                errorStruct.message = 'fluctuations must be numeric!';
                errorStruct.identifier = 'forward:NoNumbers';
                error( errorStruct );
            end
% TODO: compute rx signals for active elements for unique frequencies

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing Born approximation of the recorded RF voltage signals (kappa)...', str_date_time );

            % detect occupied grid points
            indices_occupied = find( abs( fluctuations ) > eps );

            % specify cell array for u_M
            u_M = cell( size( operator_born.discretization.spectral ) );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % compute prefactors
                %----------------------------------------------------------
                % number of unique frequencies
                N_samples_f_unique = abs( operator_born.discretization.spectral( index_measurement ).axis_k_tilde_unique );

                % extract occupied grid points from incident pressure
                p_incident_occupied = subsample( operator_born.incident_waves( index_measurement ).p_incident, [], indices_occupied );

                % extract impulse responses of mixing channels
                impulse_responses_rx = reshape( [ operator_born.discretization.spectral( index_measurement ).rx.impulse_responses ], size( operator_born.discretization.spectral( index_measurement ).rx ) );

                % extract prefactors for all mixes (current frequencies)
                indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;
                prefactors = subsample( operator_born.discretization.prefactors( index_measurement ), indices_f_to_unique );
% TODO: compute elsewhere
                prefactors = prefactors .* impulse_responses_rx;

                % map indices of the active elements to unique indices
% TODO: map unique active elements to current active elements
                indices_active_rx_to_unique = operator_born.discretization.spectral( index_measurement ).indices_active_rx_to_unique;

                % frequency axes of all mixes
% TODO: check if mixes have identical frequency axes?
                axes_f = reshape( [ impulse_responses_rx.axis ], size( operator_born.discretization.spectral( index_measurement ).rx ) );
                N_samples_f = abs( axes_f );

                %----------------------------------------------------------
                % compute mixed voltage signals for the active array elements
                %----------------------------------------------------------
                % specify cell arrays for u_M{ index_measurement }
                u_M{ index_measurement } = cell( size( operator_born.discretization.spectral( index_measurement ).rx ) );

                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    %------------------------------------------------------
                    % active array elements and pressure field (current frequencies)
                    %------------------------------------------------------
                    % number of active array elements
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    if isequal( indices_f_to_unique{ index_mix }, (1:N_samples_f_unique)' )
                        % no subsampling required
                        p_incident_occupied_act = p_incident_occupied;
                    else
                        % subsample
% TODO: optimize subsampling: do nothing if number of indices equals axes?
                        p_incident_occupied_act = subsample( p_incident_occupied, indices_f_to_unique{ index_mix } );
                    end

                    %------------------------------------------------------
                    % compute voltage signals received by the active array elements
                    %------------------------------------------------------
                    % initialize voltages with zeros
                    u_act = physical_values.voltage( zeros( N_elements_active, N_samples_f( index_mix ) ) );

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % index of active array element
                        index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( indices_occupied, indices_active_rx_to_unique{ index_mix }( index_active ) );
%                             h_rx = shift_lateral( operator_born.discretization.h_ref( index_measurement ), operator_born.discretization.spatial, index_element, indices_occupied );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref( index_measurement ).samples( indices_occupied_act, indices_f_to_unique{ index_mix } );
%                             h_rx = subsample( operator_born.discretization.h_ref( index_measurement ), indices_f_to_unique{ index_mix }, indices_occupied_act );

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % spatial transfer function of the active array element
% TODO: computes for unique frequencies?
                            h_rx = discretizations.spatial_transfer_function( operator_born.discretization.spatial, operator_born.discretization.spectral( index_measurement ), index_element );

                        end % if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % avoid spatial aliasing
                        %--------------------------------------------------
                        if operator_born.options.spatial_aliasing == scattering.options_aliasing.exclude
%                             indicator_aliasing = flag > real( axes_k_tilde( index_f ) );
%                             indicator_aliasing = indicator_aliasing .* ( 1 - ( real( axes_k_tilde( index_f ) ) ./ flag).^2 );
                        end

                        %--------------------------------------------------
                        % compute matrix-vector product
                        %--------------------------------------------------
% TODO: compute product p_incident_occupied_act .* prefactors_act outside of loop
%                         Phi_act = h_rx .* p_incident_occupied_act .* prefactors( index_mix );
                        Phi_act = double( h_rx ) .* double( p_incident_occupied_act.samples ) .* double( prefactors( index_mix ).samples( index_active, : ) );
                        u_act( index_active, : ) = fluctuations( indices_occupied ).' * Phi_act;

                    end % for index_active = 1:N_elements_active

                    %------------------------------------------------------
                    % mix voltage signals
                    %------------------------------------------------------
                    u_M{ index_measurement }{ index_mix } = sum( u_act, 1 );

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                %----------------------------------------------------------
                % create signal matrix or signals
                %----------------------------------------------------------
                if isequal( axes_f.members )
                    % create signal matrix for equal frequency axes
                    u_M{ index_measurement } = discretizations.signal_matrix( axes_f( 1 ), cat( 1, u_M{ index_measurement }{ : } ) );
                else
                    % create signals for unequal frequency axes
                    u_M{ index_measurement } = discretizations.signal( axes_f, u_M{ index_measurement } );
                end

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

            % avoid cell array for single pulse-echo measurement
            if isscalar( u_M )
                u_M = u_M{ 1 };
            end

        end % function u_M = forward( operator_born, fluctuations )

        %------------------------------------------------------------------
        % adjoint scattering (overload adjoint function)
        %------------------------------------------------------------------
        function fluctuations_hat = adjoint( operator_born, u_M )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for u_M
            if ~iscell( u_M )
                u_M = { u_M };
            end

            % initialize fluctuations_hat
            fluctuations_hat = zeros( operator_born.discretization.spatial.grid_FOV.N_points, 1 );

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % compute prefactors
                %----------------------------------------------------------
                % number of unique frequencies
                N_samples_f_unique = abs( operator_born.discretization.spectral( index_measurement ).axis_k_tilde_unique );

                % impulse responses underlying all mixed voltage signals
                impulse_responses_rx = reshape( [ operator_born.discretization.spectral( index_measurement ).rx.impulse_responses ], size( operator_born.discretization.spectral( index_measurement ).rx ) );

                % extract prefactors for all mixes (current frequencies)
                indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;
                prefactors = subsample( operator_born.discretization.prefactors( index_measurement ), indices_f_to_unique );
                prefactors = prefactors .* impulse_responses_rx;

                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    % number of active array elements
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field and prefactors for current frequencies
                    if isequal( indices_f_to_unique{ index_mix }, (1:N_samples_f_unique)' )
                        % no subsampling required
                        p_incident_act = operator_born.incident_waves( index_measurement ).p_incident;
                    else
                        % subsample
% TODO: optimize subsampling: do nothing if number of indices equals axes?
                        p_incident_act = subsample( operator_born.incident_waves( index_measurement ).p_incident, indices_f_to_unique{ index_mix } );
                    end

                    % initialize fluctuations_hat_act with zeros
                    fluctuations_hat_act = zeros( operator_born.discretization.spatial.grid_FOV.N_points, N_elements_active );

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % index of active array element
                        index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            indices_occupied_act = shift_lateral( operator_born.discretization.spatial, index_element );
                            % shift reference spatial transfer function to infer that of the active array element
%                             h_rx = shift_lateral( operator_born.discretization.h_ref( index_measurement ), operator_born.discretization.spatial, index_element );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref( index_measurement ).samples( indices_occupied_act, indices_f_to_unique{ index_mix } );
%                             h_rx = subsample( operator_born.discretization.h_ref( index_measurement ), indices_f_to_unique{ index_mix }, indices_occupied_act );

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % spatial transfer function of the active array element
                            h_rx = discretizations.spatial_transfer_function( operator_born.discretization.spatial, operator_born.discretization.spectral( index_measurement ), index_element );

                        end % if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % avoid spatial aliasing
                        %--------------------------------------------------
                        if operator_born.options.spatial_aliasing == scattering.options_aliasing.exclude
%                             indicator_aliasing = flag > real( axes_k_tilde( index_f ) );
%                             indicator_aliasing = indicator_aliasing .* ( 1 - ( real( axes_k_tilde( index_f ) ) ./ flag).^2 );
                        end

                        %--------------------------------------------------
                        % compute matrix-vector product
                        %--------------------------------------------------
                        Phi_act = double( h_rx ) .* double( p_incident_act.samples ) .* double( prefactors( index_mix ).samples( index_active, : ) );
%                         Phi_act = h_rx .* p_incident_act .* prefactors( index_mix );
                        fluctuations_hat_act( :, index_active ) = conj( Phi_act ) * u_M{ index_measurement }.samples( index_mix, : ).';

                    end % for index_active = 1:N_elements_active

                    %------------------------------------------------------
                    % superimpose contributions
                    %------------------------------------------------------
                    fluctuations_hat = fluctuations_hat + sum( fluctuations_hat_act, 2 );
                    figure(1);imagesc( illustration.dB( squeeze( reshape( double( abs( fluctuations_hat ) ), operator_born.discretization.spatial.grid_FOV.N_points_axis ) ), 20 ), [ -60, 0 ] );

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

        end % function fluctuations_hat = adjoint( operator_born, u_M )

        %------------------------------------------------------------------
        % received energy (overload energy_rx function)
        %------------------------------------------------------------------
%         function E_rx = energy_rx( operators )
% 
%             % check if calculation is required
% %             if indicator_kappa_exists_id
% % 
% %                 % load norms of column vectors for gamma_kappa
% %                 fprintf( '%s: loading norms of column vectors (gamma_%s, linear transform: %s, %.2f MByte)...', str_date_time, 'kappa', 'none', size_bytes_norms_kappa / BYTES_PER_MEBIBYTE );
% %                 load( str_filename_column_norms_id, 'norms_cols_kappa' );
% % 
% %             else
% 
%                 % compute norms of column vectors for gamma_kappa
% %                 fprintf( '%s: computing norms of column vectors (gamma_%s, linear transform: %s, %.2f MByte)...', str_date_time, 'kappa', 'none', size_bytes_norms_kappa / BYTES_PER_MEBIBYTE );
% 
%                 % compute auxiliary variables
%                 e_r_x = repmat(lattice_pos_x_elements_virtual_rx(:), [1, FOV_N_points]) - repmat(FOV_pos_X(:)', [XDC_N_points, 1]);
%                 e_r_z = -repmat(FOV_pos_Z(:)', [XDC_N_points, 1]);
%                 D_full_rx = sqrt((e_r_x).^2 + e_r_z.^2);
% 
%                 % allocate memory
%                 norms_cols_kappa = zeros( 1, FOV_N_points );
% 
%                 for index_f = 1:N_samples_f
% 
%                     hankel_0 = besselh( 0, 2, axis_k_tilde(index_f) * D_full_rx );
% 
%                     hankel_0_abs_squared_sum = zeros(XDC_N_elements, FOV_N_points);
%                     for index_element_virtual = 1:XDC_N_points
%                         hankel_0_abs_squared_sum(lattice_pos_x_elements_virtual_rx_association(index_element_virtual), :) = hankel_0_abs_squared_sum(lattice_pos_x_elements_virtual_rx_association(index_element_virtual), :) + hankel_0(index_element_virtual, :);
%                     end
%                     hankel_0_abs_squared_sum = sum( abs(hankel_0_abs_squared_sum).^2, 1 );
% 
%                     temp_kappa = abs( prefactor_kappa(index_f) ).^2 * hankel_0_abs_squared_sum;
% 
%                     temp_sum_p_in = zeros(1, FOV_N_points);
%                     for index_incident = 1:N_incident
% 
%                         temp_sum_p_in = temp_sum_p_in + (abs( p_incident{index_f, index_incident}(:) ).^2)';
%                     end
% 
%                     norms_cols_kappa = norms_cols_kappa + temp_kappa .* temp_sum_p_in;
% 
%                 end % for index_f = 1:N_samples_f
% 
%                 norms_cols_kappa = sqrt( norms_cols_kappa )';
% 
%                 % save results to disk
% %                 if indicator_file_exists_id
% %                     % append result to existing file
% %                     save( str_filename_column_norms_id, 'norms_cols_kappa', '-append' );
% %                 else
% %                     % save result to new file
% %                     save( str_filename_column_norms_id, 'norms_cols_kappa', '-v7.3' );
% %                     indicator_file_exists_id = 1;
% %                 end
% %             end % if indicator_kappa_exists_id
% 
%         end % function E_rx = energy_rx( operators )

        %------------------------------------------------------------------
        % matrix multiplication (overload mtimes function)
        %------------------------------------------------------------------
        function u_M = mtimes( operator_born, fluctuations )

            %--------------------------------------------------------------
            % 1.) call forward scattering
            %--------------------------------------------------------------
            u_M = forward( operator_born, fluctuations );

        end % function u_M = mtimes( operator_born, fluctuations )

	end % methods

end % classdef operator_born < scattering.operator
