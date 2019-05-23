%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-05-11
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
        % quick forward scattering
        %------------------------------------------------------------------
        function u_M = forward_quick( operator_born, fluctuations, varargin )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: quick forward scattering (Born approximation, kappa)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure numeric matrix
            if ~( isnumeric( fluctuations ) && ismatrix( fluctuations ) )
                errorStruct.message = 'fluctuations must be a numeric matrix!';
                errorStruct.identifier = 'forward_quick:NoNumericMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) adjoint linear transform
            %--------------------------------------------------------------
% TODO: check compatibility  && isequal( operator_born.discretization.spatial.grid_FOV.N_points_axis, varargin{ 1 }.N_lattice )
            if nargin >= 3 && isa( varargin{ 1 }, 'linear_transforms.linear_transform' )
                % apply adjoint linear transform
                fluctuations = operator_transform( varargin{ 1 }, fluctuations, 2 );
            end

            %--------------------------------------------------------------
            % 3.) compute mixed voltage signals
            %--------------------------------------------------------------
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
%                 indices_active_rx_to_unique = operator_born.discretization.spectral( index_measurement ).indices_active_rx_to_unique;

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
                    u_M{ index_measurement }{ index_mix } = zeros( N_objects, N_samples_f( index_mix ) );

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
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( indices_occupied, index_element );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref.samples( indices_occupied_act, indices_f_to_unique{ index_mix } );

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
                        % compute matrix-vector product and mix voltage signals
                        %--------------------------------------------------
                        Phi_act = double( h_rx ) .* p_incident_occupied_act .* double( prefactors( index_mix ).samples( index_active, : ) );
                        u_M{ index_measurement }{ index_mix } = u_M{ index_measurement }{ index_mix } + fluctuations( indices_occupied, : ).' * Phi_act;

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                % concatenate cell arrays into matrix
                u_M{ index_measurement } = cat( 2, u_M{ index_measurement }{ : } );

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

        	% concatenate cell arrays into matrix
            u_M = cat( 2, u_M{ : } ).';

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function u_M = forward_quick( operator_born, fluctuations )

        %------------------------------------------------------------------
        % forward scattering (overload forward function)
        %------------------------------------------------------------------
        function u_M = forward( operator_born, fluctuations, varargin )
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
%                 indices_active_rx_to_unique = operator_born.discretization.spectral( index_measurement ).indices_active_rx_to_unique;

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

                        % index of active array element
                        index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( indices_occupied, index_element );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref.samples( indices_occupied_act, indices_f_to_unique{ index_mix } );

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
        % quick adjoint scattering
        %------------------------------------------------------------------
        function theta_hat = adjoint_quick( operator_born, u_M, varargin )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: quick adjoint scattering (Born approximation, kappa)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure numeric matrix
            if ~( isnumeric( u_M ) && ismatrix( u_M ) )
                errorStruct.message = 'u_M must be a numeric matrix!';
                errorStruct.identifier = 'adjoint_quick:NoNumericMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute adjoint fluctuations
            %--------------------------------------------------------------
            % initialize theta_hat
            theta_hat = zeros( operator_born.discretization.spatial.grid_FOV.N_points, size( u_M, 2 ) );

            % partition matrix into cell arrays
            u_M = mat2cell( u_M, cellfun( @( x ) sum( x( : ) ), { operator_born.discretization.spectral.N_observations } ), size( u_M, 2 ) );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % compute prefactors
                %----------------------------------------------------------
                % map unique frequencies of pulse-echo measurements to unique frequencies
                indices_f_to_unique_measurement = operator_born.discretization.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies
                indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.discretization.prefactors{ index_measurement };

                % partition matrix into cell arrays
                u_M{ index_measurement } = mat2cell( u_M{ index_measurement }, operator_born.discretization.spectral( index_measurement ).N_observations, size( u_M{ index_measurement }, 2 ) );

                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    % number of active array elements
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    p_incident_act = double( operator_born.incident_waves( index_measurement ).p_incident.samples );
                    if numel( indices_f_to_unique{ index_mix } ) < abs( operator_born.incident_waves( index_measurement ).p_incident.axis )
                        p_incident_act = p_incident_act( :, indices_f_to_unique{ index_mix } );
                    end

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
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( :, index_element );

                            % extract current frequencies from unique frequencies
                            if numel( indices_f_to_unique{ index_mix } ) < abs( operator_born.discretization.h_ref.axis )
                                h_rx = double( operator_born.discretization.h_ref.samples( indices_occupied_act, indices_f_to_unique_measurement( indices_f_to_unique{ index_mix } ) ) );
                            else
                                h_rx = double( operator_born.discretization.h_ref.samples( indices_occupied_act, : ) );
                            end

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
% TODO: precompute at appropriate location
%                             indicator_aliasing = flag > real( axes_k_tilde( index_f ) );
%                             indicator_aliasing = indicator_aliasing .* ( 1 - ( real( axes_k_tilde( index_f ) ) ./ flag).^2 );
                        end

                        %--------------------------------------------------
                        % compute matrix-vector product
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_act .* double( prefactors( index_mix ).samples( index_active, : ) );
                        theta_hat = theta_hat + conj( Phi_act ) * u_M{ index_measurement }{ index_mix };

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

            %--------------------------------------------------------------
            % 3.) forward linear transform
            %--------------------------------------------------------------
            if nargin >= 3 && isa( varargin{ 1 }, 'linear_transforms.linear_transform' )
                % apply forward linear transform
                theta_hat = operator_transform( varargin{ 1 }, theta_hat, 1 );
            end

            % illustrate
            figure(999);
            imagesc( illustration.dB( squeeze( reshape( double( abs( theta_hat( :, 1 ) ) ), operator_born.discretization.spatial.grid_FOV.N_points_axis ) ), 20 ), [ -60, 0 ] );

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function theta_hat = adjoint_quick( operator_born, u_M, varargin )

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
%                 indices_active_rx_to_unique = operator_born.discretization.spectral( index_measurement ).indices_active_rx_to_unique;

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

                        % index of active array element
                        index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( :, index_element );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref.samples( indices_occupied_act, indices_f_to_unique{ index_mix } );

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
        % combined
        %------------------------------------------------------------------
        function y = combined( operator_born, x, mode, varargin )

            switch mode

                case 0
                    % return size of forward transform
                    y = operator_born.discretization.size;
                case 1
                    % quick forward scattering
                    y = forward_quick( operator_born, x, varargin{ : } );
                case 2
                    % quick adjoint scattering
                    y = adjoint_quick( operator_born, x, varargin{ : } );
                otherwise
                    % unknown operation
                    errorStruct.message = 'Unknown mode of operation!';
                    errorStruct.identifier = 'combined:InvalidMode';
                    error( errorStruct );

            end % switch mode

        end % function y = combined( operator_born, x, mode, varargin )

        %------------------------------------------------------------------
        % transform point spread function (overload tpsf function)
        %------------------------------------------------------------------
        function [ theta_hat, E_rx, adjointness ] = tpsf( operators_born, indices, varargin )

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
            % 2.) compute TPSFs
            %--------------------------------------------------------------
            % specify cell array for tpsf
            theta_hat = cell( size( operators_born ) );
            E_rx = cell( size( operators_born ) );
            adjointness = cell( size( operators_born ) );

            % iterate scattering operators
            for index_object = 1:numel( operators_born )

                % numbers of TPSFs and grid points
                N_tpsf = numel( indices{ index_object } );
                N_points = operators_born( index_object ).discretization.spatial.grid_FOV.N_points;

                % indices of coefficients
                indices_tpsf = (0:N_tpsf-1) * N_points + indices{ index_object };

                % initialize coefficient vectors
                theta = zeros( N_points, N_tpsf );
                theta( indices_tpsf ) = 1;

                % quick forward scattering and received energies
                u_M = forward_quick( operators_born( index_object ), theta, varargin{ : } );
                E_rx{ index_object } = vecnorm( u_M, 2, 1 ).^2;

                % quick adjoint scattering and test for adjointness
                theta_hat{ index_object } = adjoint_quick( operators_born( index_object ), u_M, varargin{ : } );
                adjointness{ index_object } = E_rx{ index_object } - theta_hat{ index_object }( indices_tpsf );

            end % for index_object = 1:numel( operators_born )

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                theta_hat = theta_hat{ 1 };
                E_rx = E_rx{ 1 };
                adjointness = adjointness{ 1 };
            end

        end % function [ theta_hat, E_rx, adjointness ] = tpsf( operators_born, indices, varargin )

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
                % map unique frequencies of pulse-echo measurements to unique frequencies
                indices_f_to_unique_measurement = operator_born.discretization.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies
                indices_f_to_unique = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

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

                        % index of active array element
                        index_element = operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.discretization.indices_grid_FOV_shift( :, index_element );

                            % extract current frequencies from unique frequencies
                            h_rx = operator_born.discretization.h_ref.samples( indices_occupied_act, indices_f_to_unique_measurement( indices_f_to_unique{ index_mix } ) );

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
            %----------------------------------------------------------
            E_M = physical_values.volt( E_M ) * physical_values.volt;

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_M = energy_rx( operator_born )

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
