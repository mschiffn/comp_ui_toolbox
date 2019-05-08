%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-05-06
%
classdef operator_born < scattering.operator

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % dependent properties
        

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
        % forward scattering (overload forward function)
        %------------------------------------------------------------------
        function u_M = forward( operator_born, fluctuations )
% TODO: compute rx signals for active elements for unique frequencies

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if ~( isnumeric( fluctuations ) && ismatrix( fluctuations ) )
                errorStruct.message = 'fluctuations must be a numeric matrix!';
                errorStruct.identifier = 'forward:NoNumericMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute mixed voltage signals
            %--------------------------------------------------------------
            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing Born approximation of the recorded RF voltage signals (kappa)...', str_date_time );

            % detect occupied grid points
            N_objects = size( fluctuations, 2 );
            indices_occupied = find( sum( abs( fluctuations ), 2 ) > eps );

            % specify cell array for u_M
            u_M = cell( size( operator_born.discretization.spectral ) );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % prefactors
                %----------------------------------------------------------
                % map frequencies of mixed voltage signals to unique frequencies
                indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

                % map indices of the active elements to unique indices
                indices_active_rx_to_unique = operator_born.discretization.spectral( index_measurement ).indices_active_rx_to_unique;

                % extract occupied grid points from incident pressure
                p_incident_occupied = double( operator_born.incident_waves( index_measurement ).p_incident.samples( indices_occupied, : ) );

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.discretization.prefactors{ index_measurement };

                % numbers of frequencies in mixed voltage signals
                axes_f = reshape( [ prefactors.axis ], size( prefactors ) );
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
                    p_incident_occupied_act = p_incident_occupied( :, indices_f_to_unique{ index_mix } );

                    %------------------------------------------------------
                    % compute voltage signals received by the active array elements
                    %------------------------------------------------------
                    % initialize voltages with zeros
                    u_mix = zeros( N_objects, N_samples_f( index_mix ) );

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( indices_occupied, indices_active_rx_to_unique{ index_mix }( index_active ) );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref( index_measurement ).samples( indices_occupied_act, indices_f_to_unique{ index_mix } );

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % index of active array element
                            index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

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
                        % compute matrix-vector product and mix voltage signals
                        %--------------------------------------------------
                        Phi_act = double( h_rx ) .* p_incident_occupied_act .* double( prefactors( index_mix ).samples( index_active, : ) );
                        u_mix = u_mix + fluctuations( indices_occupied, : ).' * Phi_act;

                    end % for index_active = 1:N_elements_active

                    %------------------------------------------------------
                    % 
                    %------------------------------------------------------
                    u_M{ index_measurement }{ index_mix } = u_mix;

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                %----------------------------------------------------------
                % create signal matrix or signals
                %----------------------------------------------------------
%                 if isequal( axes_f.members )
%                     % create signal matrix for equal frequency axes
%                     u_M{ index_measurement } = discretizations.signal_matrix( axes_f( 1 ), cat( 1, u_M{ index_measurement }{ : } ) );
%                 else
%                     % create signals for unequal frequency axes
%                     u_M{ index_measurement } = discretizations.signal( axes_f, u_M{ index_measurement } );
%                 end

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

            %--------------------------------------------------------------
            % 2.) compute adjoint fluctuations
            %--------------------------------------------------------------
            % initialize fluctuations_hat
            fluctuations_hat = zeros( operator_born.discretization.spatial.grid_FOV.N_points, 1 );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % compute prefactors
                %----------------------------------------------------------
                % map frequencies of mixed voltage signals to unique frequencies
                indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

                % map indices of the active elements to unique indices
                indices_active_rx_to_unique = operator_born.discretization.spectral( index_measurement ).indices_active_rx_to_unique;

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.discretization.prefactors{ index_measurement };

                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    % number of active array elements
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    p_incident_act = double( operator_born.incident_waves( index_measurement ).p_incident.samples( :, indices_f_to_unique{ index_mix } ) );

                    % initialize fluctuations_hat_act with zeros
                    fluctuations_hat_act = zeros( operator_born.discretization.spatial.grid_FOV.N_points, N_elements_active );

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( :, indices_active_rx_to_unique{ index_mix }( index_active ) );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref( index_measurement ).samples( indices_occupied_act, indices_f_to_unique{ index_mix } );

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % index of active array element
                            index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

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
                        Phi_act = double( h_rx ) .* p_incident_act .* double( prefactors( index_mix ).samples( index_active, : ) );
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
        % point spread function (overload psf function)
        %------------------------------------------------------------------
        function [ out, E_rx, adjointness ] = psf( operators_born, indices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, indices );

            %--------------------------------------------------------------
            % 2.) compute PSFs
            %--------------------------------------------------------------
            % specify cell array for psf
            out = cell( size( operators_born ) );
            E_rx = cell( size( operators_born ) );
            adjointness = cell( size( operators_born ) );

            % iterate scattering operators
            for index_object = 1:numel( operators_born )

                % number of PSFs
                N_psf = numel( indices{ index_object } );

                % initialize coefficient vectors and output with zeros
                out{ index_object } = zeros( operators_born( index_object ).discretization.spatial.grid_FOV.N_points, N_psf );

                % iterate sequential pulse-echo measurements
                for index_measurement = 1:numel( operators_born( index_object ).discretization.spectral )

                    %------------------------------------------------------
                    % compute prefactors
                    %------------------------------------------------------
                    % extract impulse responses of mixing channels
                    impulse_responses_rx = reshape( [ operators_born( index_object ).discretization.spectral( index_measurement ).rx.impulse_responses ], size( operators_born( index_object ).discretization.spectral( index_measurement ).rx ) );

                    % extract prefactors for all mixes (current frequencies)
                    indices_f_to_unique = operators_born( index_object ).discretization.spectral( index_measurement ).indices_f_to_unique;
                    prefactors = subsample( operators_born( index_object ).discretization.prefactors( index_measurement ), indices_f_to_unique );
% TODO: compute elsewhere
                    prefactors = prefactors .* impulse_responses_rx;

                    % map indices of the active elements to unique indices
                    indices_active_rx_to_unique = operators_born( index_object ).discretization.spectral( index_measurement ).indices_active_rx_to_unique;

                    % iterate mixed voltage signals
                    for index_mix = 1:numel( operators_born( index_object ).discretization.spectral( index_measurement ).rx )

                        % number of active array elements
                        N_elements_active = numel( operators_born( index_object ).discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                        % extract incident acoustic pressure field for current frequencies
                        p_incident_act = double( operators_born( index_object ).incident_waves( index_measurement ).p_incident.samples( :, indices_f_to_unique{ index_mix } ) );

                        % initialize fluctuations_hat_act with zeros
                        fluctuations_hat_act = zeros( operators_born( index_object ).discretization.spatial.grid_FOV.N_points, numel( indices{ index_object } ) );

                        % iterate active array elements
                        for index_active = 1:N_elements_active

                            % spatial transfer function of the active array element
                            if isa( operators_born( index_object ).discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                                %------------------------------------------
                                % a) symmetric spatial discretization based on orthogonal regular grids
                                %------------------------------------------
                                % shift reference spatial transfer function to infer that of the active array element
                                indices_occupied_act = operators_born( index_object ).discretization.indices_grid_FOV_shift( :, indices_active_rx_to_unique{ index_mix }( index_active ) );

                                % extract current frequencies from unique frequencies
                                h_rx = operators_born( index_object ).discretization.h_ref( index_measurement ).samples( indices_occupied_act, indices_f_to_unique{ index_mix } );

                            else

                                %------------------------------------------
                                % b) arbitrary grid
                                %------------------------------------------
                                % index of active array element
                                index_element = operators_born( index_object ).discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                                % spatial transfer function of the active array element
                                h_rx = discretizations.spatial_transfer_function( operators_born( index_object ).discretization.spatial, operators_born( index_object ).discretization.spectral( index_measurement ), index_element );

                            end % if isa( operators_born( index_object ).discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % avoid spatial aliasing
                            %----------------------------------------------
                            if operators_born( index_object ).options.spatial_aliasing == scattering.options_aliasing.exclude
%                               indicator_aliasing = flag > real( axes_k_tilde( index_f ) );
%                               indicator_aliasing = indicator_aliasing .* ( 1 - ( real( axes_k_tilde( index_f ) ) ./ flag).^2 );
                            end

                            %----------------------------------------------
                            % compute matrix-vector products
                            %----------------------------------------------
                            Phi_act = double( h_rx ) .* p_incident_act .* double( prefactors( index_mix ).samples( index_active, : ) );
                            fluctuations_hat_act = fluctuations_hat_act + conj( Phi_act ) * Phi_act( indices{ index_object }, : ).';

                        end % for index_active = 1:N_elements_active

                        %--------------------------------------------------
                        % superimpose contributions
                        %--------------------------------------------------
                        fluctuations_hat = fluctuations_hat + sum( fluctuations_hat_act, 2 );
                        figure(1);imagesc( illustration.dB( squeeze( reshape( double( abs( fluctuations_hat ) ), operators_born( index_object ).discretization.spatial.grid_FOV.N_points_axis ) ), 20 ), [ -60, 0 ] );

                    end % for index_mix = 1:numel( operators_born( index_object ).discretization.spectral( index_measurement ).rx )

                end % for index_measurement = 1:numel( operators_born( index_object ).discretization.spectral )

            end % for index_object = 1:numel( operators_born )

        end % function [ out, E_rx, adjointness ] = psf( operators_born, indices )

        %------------------------------------------------------------------
        % received energy (overload energy_rx function)
        %------------------------------------------------------------------
        function E_M = energy_rx( operator_born )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing received energies (kappa)...', str_date_time );

            % initialize received energies with zeros
            E_M = zeros( operator_born.discretization.spatial.grid_FOV.N_points, 1 );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % prefactors
                %----------------------------------------------------------
                % map frequencies of mixed voltage signals to unique frequencies
                indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

                % map indices of the active elements to unique indices
                indices_active_rx_to_unique = operator_born.discretization.spectral( index_measurement ).indices_active_rx_to_unique;

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.discretization.prefactors{ index_measurement };

                % numbers of frequencies in mixed voltage signals
                axes_f = reshape( [ prefactors.axis ], size( prefactors ) );
                N_samples_f = abs( axes_f );

                %----------------------------------------------------------
                % compute mixed voltage signals for the active array elements
                %----------------------------------------------------------
                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    %------------------------------------------------------
                    % active array elements and pressure field (current frequencies)
                    %------------------------------------------------------
                    % number of active array elements
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    p_incident_act = double( operator_born.incident_waves( index_measurement ).p_incident.samples( :, indices_f_to_unique{ index_mix } ) );

                    %------------------------------------------------------
                    % compute voltage signals received by the active array elements
                    %------------------------------------------------------
                    % initialize voltages with zeros
                    Phi_M = zeros( operator_born.discretization.spatial.grid_FOV.N_points, N_samples_f( index_mix ) );

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( :, indices_active_rx_to_unique{ index_mix }( index_active ) );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref( index_measurement ).samples( indices_occupied_act, indices_f_to_unique{ index_mix } );

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % index of active array element
                            index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

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
                        % compute energies
                        %--------------------------------------------------
                        Phi_act = double( h_rx ) .* p_incident_act .* double( prefactors( index_mix ).samples( index_active, : ) );
                        Phi_M = Phi_M + Phi_act;

                    end % for index_active = 1:N_elements_active

                    %------------------------------------------------------
                    % compute energy of mixed voltage signals
                    %------------------------------------------------------
                    E_M = E_M + vecnorm( Phi_M, 2, 2 ).^2;

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

            %----------------------------------------------------------
            % create signal matrix or signals
            %----------------------------------------------------------
            E_M = physical_values.voltage( E_M ) * physical_values.voltage;

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_rx = energy_rx( operators_born )

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
