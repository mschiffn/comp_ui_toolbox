%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-07-11
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
        % quick forward scattering (wrapper)
        %------------------------------------------------------------------
        function u_M = forward_quick( operator_born, fluctuations, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born (scalar)
            if ~( isa( operator_born, 'scattering.operator_born' ) && isscalar( operator_born ) )
                errorStruct.message = 'operator_born must be a single scattering.operator_born!';
                errorStruct.identifier = 'forward_quick:NoSingleOperatorBorn';
                error( errorStruct );
            end

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
            if nargin >= 3 && isscalar( varargin{ 1 } ) && isa( varargin{ 1 }, 'linear_transforms.linear_transform' )
                % apply adjoint linear transform
                fluctuations = operator_transform( varargin{ 1 }, fluctuations, 2 );
            end

            %--------------------------------------------------------------
            % 3.) compute mixed voltage signals
            %--------------------------------------------------------------
            if operator_born.options.momentary.gpu.status == scattering.options_gpu_status.on
% TODO: remove complex
                u_M = scattering.combined_quick_gpu( operator_born, 1, complex( fluctuations ) );
%                 clear mex;
            else
                u_M = forward_quick_cpu( operator_born, fluctuations );
            end

        end % function u_M = forward_quick( operator_born, fluctuations, varargin )

        %------------------------------------------------------------------
        % quick forward scattering (CPU)
        %------------------------------------------------------------------
        function u_M = forward_quick_cpu( operator_born, fluctuations )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: quick forward scattering (CPU, Born approximation, double precision, kappa)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born (scalar)
            if ~( isa( operator_born, 'scattering.operator_born' ) && isscalar( operator_born ) )
                errorStruct.message = 'operator_born must be a single scattering.operator_born!';
                errorStruct.identifier = 'forward_quick_cpu:NoSingleOperatorBorn';
                error( errorStruct );
            end

            % ensure numeric matrix
            if ~( isnumeric( fluctuations ) && ismatrix( fluctuations ) )
                errorStruct.message = 'fluctuations must be a numeric matrix!';
                errorStruct.identifier = 'forward_quick_cpu:NoNumericMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute mixed voltage signals
            %--------------------------------------------------------------
            % detect occupied grid points
            N_objects = size( fluctuations, 2 );
            indices_occupied = find( sum( abs( fluctuations ), 2 ) > eps );

            % specify cell array for u_M
            u_M = cell( size( operator_born.discretization.spectral ) );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % i.) frequency maps, prefactors, and frequency axes
                %----------------------------------------------------------
                % map unique frequencies of pulse-echo measurement to global unique frequencies
                indices_f_measurement_to_global = operator_born.discretization.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                indices_f_mix_to_measurement = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

                % extract occupied grid points from incident pressure
                p_incident_occupied = double( operator_born.incident_waves( index_measurement ).p_incident.samples( :, indices_occupied ) );

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.discretization.prefactors{ index_measurement };

                % numbers of frequencies in mixed voltage signals
                axes_f = reshape( [ prefactors.axis ], size( prefactors ) );
                N_samples_f = abs( axes_f );

                %----------------------------------------------------------
                % ii.) compute mixed voltage signals for the active array elements
                %----------------------------------------------------------
                % specify cell arrays for u_M{ index_measurement }
                u_M{ index_measurement } = cell( size( operator_born.discretization.spectral( index_measurement ).rx ) );

                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    %------------------------------------------------------
                    % a) active array elements and pressure field (current frequencies)
                    %------------------------------------------------------
                    % number of active array elements
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    p_incident_occupied_act = p_incident_occupied( indices_f_mix_to_measurement{ index_mix }, : );

                    %------------------------------------------------------
                    % b) compute voltage signals received by the active array elements
                    %------------------------------------------------------
                    % initialize mixed voltage signals with zeros
                    u_M{ index_measurement }{ index_mix } = zeros( N_samples_f( index_mix ), N_objects );

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
                            indices_occupied_act = operator_born.discretization.spatial.indices_grid_FOV_shift( indices_occupied, index_element );

                            % extract current frequencies from unique frequencies
                            if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.discretization.h_ref.axis )

                                if operator_born.options.momentary.anti_aliasing.status == scattering.options_anti_aliasing_status.on
                                    h_rx = double( operator_born.h_ref_aa.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                                else
                                    h_rx = double( operator_born.discretization.h_ref.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                                end

                            else

                                if operator_born.options.momentary.anti_aliasing.status == scattering.options_anti_aliasing_status.on
                                    h_rx = double( operator_born.h_ref_aa.samples( :, indices_occupied_act ) );
                                else
                                    h_rx = double( operator_born.discretization.h_ref.samples( :, indices_occupied_act ) );
                                end

                            end

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % compute spatial transfer function of the active array element
                            h_rx = discretizations.spatial_transfer_function( operator_born.discretization.spatial, axes_f( index_mix ), index_element );
                            h_rx = double( h_rx.samples );

                            % apply spatial anti-aliasing filter
                            
                        end % if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % compute matrix-vector product and mix voltage signals
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_occupied_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        u_M{ index_measurement }{ index_mix } = u_M{ index_measurement }{ index_mix } + Phi_act * fluctuations( indices_occupied, : );

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                % concatenate cell array contents into matrix
                u_M{ index_measurement } = cat( 1, u_M{ index_measurement }{ : } );

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

            % concatenate cell array contents into matrix
            u_M = cat( 1, u_M{ : } );

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function u_M = forward_quick_cpu( operator_born, fluctuations )

        %------------------------------------------------------------------
        % quick forward scattering (GPU: C++ & CUDA API)
        %------------------------------------------------------------------
        % see combined_quick_gpu.cu

        %------------------------------------------------------------------
        % quick adjoint scattering (wrapper)
        %------------------------------------------------------------------
        function theta_hat = adjoint_quick( operator_born, u_M, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born (scalar)
            if ~( isa( operator_born, 'scattering.operator_born' ) && isscalar( operator_born ) )
                errorStruct.message = 'operator_born must be a single scattering.operator_born!';
                errorStruct.identifier = 'adjoint_quick:NoSingleOperatorBorn';
                error( errorStruct );
            end

            % ensure numeric matrix
            if ~( isnumeric( u_M ) && ismatrix( u_M ) )
                errorStruct.message = 'u_M must be a numeric matrix!';
                errorStruct.identifier = 'adjoint_quick:NoNumericMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute adjoint fluctuations
            %--------------------------------------------------------------
            if operator_born.options.momentary.gpu.status == scattering.options_gpu_status.on
                gamma_hat = scattering.combined_quick_gpu( operator_born, 2, u_M );
%                 clear mex;
            else
                gamma_hat = adjoint_quick_cpu( operator_born, u_M );
            end

            %--------------------------------------------------------------
            % 3.) forward linear transform
            %--------------------------------------------------------------
            if nargin >= 3 && isscalar( varargin{ 1 } ) && isa( varargin{ 1 }, 'linear_transforms.linear_transform' )
                % apply forward linear transform
                theta_hat = operator_transform( varargin{ 1 }, gamma_hat, 1 );
            end

            % illustrate
            figure(999);
            subplot( 1, 2, 1 );
            imagesc( illustration.dB( squeeze( reshape( double( abs( gamma_hat( :, 1 ) ) ), operator_born.discretization.spatial.grid_FOV.N_points_axis ) ), 20 )', [ -60, 0 ] );
            subplot( 1, 2, 2 );
            imagesc( illustration.dB( squeeze( reshape( double( abs( theta_hat( :, 1 ) ) ), operator_born.discretization.spatial.grid_FOV.N_points_axis ) ), 20 )', [ -60, 0 ] );

        end % function theta_hat = adjoint_quick( operator_born, u_M, varargin )

        %------------------------------------------------------------------
        % quick adjoint scattering (CPU)
        %------------------------------------------------------------------
        function gamma_hat = adjoint_quick_cpu( operator_born, u_M )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: quick adjoint scattering (CPU, Born approximation, double precision, kappa)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born (scalar)
            if ~( isa( operator_born, 'scattering.operator_born' ) && isscalar( operator_born ) )
                errorStruct.message = 'operator_born must be a single scattering.operator_born!';
                errorStruct.identifier = 'adjoint_quick_cpu:NoSingleOperatorBorn';
                error( errorStruct );
            end

            % ensure numeric matrix
            if ~( isnumeric( u_M ) && ismatrix( u_M ) )
                errorStruct.message = 'u_M must be a numeric matrix!';
                errorStruct.identifier = 'adjoint_quick_cpu:NoNumericMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute adjoint fluctuations
            %--------------------------------------------------------------
            % initialize gamma_hat
            gamma_hat = zeros( operator_born.discretization.spatial.grid_FOV.N_points, size( u_M, 2 ) );

            % partition matrix into cell arrays
            u_M = mat2cell( u_M, cellfun( @( x ) sum( x( : ) ), { operator_born.discretization.spectral.N_observations } ), size( u_M, 2 ) );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operator_born.discretization.spectral )

                %----------------------------------------------------------
                % a) frequency maps and prefactors
                %----------------------------------------------------------
                % map unique frequencies of pulse-echo measurement to global unique frequencies
                indices_f_measurement_to_global = operator_born.discretization.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                indices_f_mix_to_measurement = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

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
                    if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.incident_waves( index_measurement ).p_incident.axis )
                        p_incident_act = p_incident_act( indices_f_mix_to_measurement{ index_mix }, : );
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
                            indices_occupied_act = operator_born.discretization.spatial.indices_grid_FOV_shift( :, index_element );

                            % extract current frequencies from unique frequencies
                            if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.discretization.h_ref.axis )

                                if operator_born.options.momentary.anti_aliasing.status == scattering.options_anti_aliasing_status.on
                                    h_rx = double( operator_born.h_ref_aa.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                                else
                                    h_rx = double( operator_born.discretization.h_ref.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                                end

                            else

                                if operator_born.options.momentary.anti_aliasing.status == scattering.options_anti_aliasing_status.on
                                    h_rx = double( operator_born.h_ref_aa.samples( :, indices_occupied_act ) );
                                else
                                    h_rx = double( operator_born.discretization.h_ref.samples( :, indices_occupied_act ) );
                                end

                            end

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % compute spatial transfer function of the active array element
                            h_rx = discretizations.spatial_transfer_function( operator_born.discretization.spatial, axes_f( index_mix ), index_element );
                            h_rx = double( h_rx.samples );

                        end % if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % compute matrix-vector product
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        gamma_hat = gamma_hat + Phi_act' * u_M{ index_measurement }{ index_mix };

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

            end % for index_measurement = 1:numel( operator_born.discretization.spectral )

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function gamma_hat = adjoint_quick_cpu( operator_born, u_M )

        %------------------------------------------------------------------
        % quick adjoint scattering (GPU: C++ & CUDA API)
        %------------------------------------------------------------------
        % see combined_quick_gpu.cu

        %------------------------------------------------------------------
        % quick combined scattering
        %------------------------------------------------------------------
        function y = combined_quick( operator_born, x, mode, varargin )

            switch mode

                case 0
                    % return size of forward transform
                    y = operator_born.discretization.size;
                case 1
                    % quick forward scattering (wrapper)
                    y = forward_quick( operator_born, x, varargin{ : } );
                case 2
                    % quick adjoint scattering (wrapper)
                    y = adjoint_quick( operator_born, x, varargin{ : } );
                otherwise
                    % unknown operation
                    errorStruct.message = 'Unknown mode of operation!';
                    errorStruct.identifier = 'combined_quick:InvalidMode';
                    error( errorStruct );

            end % switch mode

        end % function y = combined_quick( operator_born, x, mode, varargin )

        %------------------------------------------------------------------
        % forward scattering (overload forward method)
        %------------------------------------------------------------------
        function u_M = forward( operators_born, fluctuations, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'forward:NoOperatorsBorn';
                error( errorStruct );
            end

            % ensure cell array for fluctuations
            if ~iscell( fluctuations )
                fluctuations = { fluctuations };
            end

            % ensure nonempty linear_transforms
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                linear_transforms = varargin{ 1 };
            else
                % empty linear_transform is identity
                linear_transforms = cell( size( operators_born ) );
            end

            % ensure cell array for linear_transforms
            if ~iscell( linear_transforms )
                linear_transforms = { linear_transforms };
            end

            % overwrite properties of momentary scattering operator options
            if nargin >= 4
                operators_born = set_properties_momentary( operators_born, varargin{ 2:end } );
            end

            % multiple operators_born / single fluctuations
            if ~isscalar( operators_born ) && isscalar( fluctuations )
                fluctuations = repmat( fluctuations, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, fluctuations, linear_transforms );

            %--------------------------------------------------------------
            % 2.) compute mixed voltage signals
            %--------------------------------------------------------------
            % specify cell array for u_M
            u_M = cell( size( operators_born ) );

            % iterate scattering operators
            for index_object = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check fluctuations and linear transform
                %----------------------------------------------------------
                % ensure numeric matrix
                if ~( isnumeric( fluctuations{ index_object } ) && ismatrix( fluctuations{ index_object } ) )
                    errorStruct.message = sprintf( 'fluctuations{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward:NoNumericMatrix';
                    error( errorStruct );
                end

                % method forward_quick ensures class linear_transforms.linear_transform

                %----------------------------------------------------------
                % b) quick adjoint scattering
                %----------------------------------------------------------
%                 profile on
                u_M{ index_object } = forward_quick( operators_born( index_object ), fluctuations{ index_object }( : ), linear_transforms{ index_object } );
                u_M{ index_object } = physical_values.volt( u_M{ index_object } );
%                 profile viewer

                %----------------------------------------------------------
                % c) create signals or signal matrices
                %----------------------------------------------------------
                % partition matrix into cell arrays
                u_M{ index_object } = mat2cell( u_M{ index_object }, cellfun( @( x ) sum( x( : ) ), { operators_born( index_object ).discretization.spectral.N_observations } ), size( u_M{ index_object }, 2 ) );

                % iterate sequential pulse-echo measurements
                for index_measurement = 1:numel( operators_born( index_object ).discretization.spectral )

                    % map unique frequencies of pulse-echo measurement to global unique frequencies
                    indices_f_measurement_to_global = operators_born( index_object ).discretization.indices_f_to_unique{ index_measurement };

                    % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                    indices_f_mix_to_measurement = operators_born( index_object ).discretization.spectral( index_measurement ).indices_f_to_unique;

                    % partition matrix into cell arrays
                    u_M{ index_object }{ index_measurement } = mat2cell( u_M{ index_object }{ index_measurement }, operators_born( index_object ).discretization.spectral( index_measurement ).N_observations, size( u_M{ index_object }{ index_measurement }, 2 ) );

                    % subsample global unique frequencies to get unique frequencies of pulse-echo measurement
                    axis_f_measurement_unique = subsample( operators_born( index_object ).discretization.axis_f_unique, indices_f_measurement_to_global );

                    % subsample unique frequencies of pulse-echo measurement to get frequencies of mixed voltage signals
                    axes_f_mix = reshape( subsample( axis_f_measurement_unique, indices_f_mix_to_measurement ), size( u_M{ index_object }{ index_measurement } ) );

                    % create mixed voltage signals
                    u_M{ index_object }{ index_measurement } = discretizations.signal( axes_f_mix, u_M{ index_object }{ index_measurement } );

                    % try to merge mixed voltage signals
                    try
                        u_M{ index_object }{ index_measurement } = merge( u_M{ index_object }{ index_measurement } );
                    catch
                    end

                end % for index_measurement = 1:numel( operators_born( index_object ).discretization.spectral )

                % create array of signal matrices
                if all( cellfun( @( x ) strcmp( class( x( : ) ), 'discretizations.signal_matrix' ), u_M{ index_object } ) )
                    u_M{ index_object } = [ u_M{ index_object }{ : } ];
                end

            end % for index_object = 1:numel( operators_born )

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                u_M = u_M{ 1 };
            end

        end % function u_M = forward( operators_born, fluctuations, varargin )

        %------------------------------------------------------------------
        % adjoint scattering (overload adjoint method)
        %------------------------------------------------------------------
        function theta_hat = adjoint( operators_born, u_M, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'adjoint:NoOperatorsBorn';
                error( errorStruct );
            end

            % ensure cell array for u_M
            if ~iscell( u_M )
                u_M = { u_M };
            end

            % ensure nonempty linear_transforms
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                linear_transforms = varargin{ 1 };
            else
                % empty linear_transform is identity
                linear_transforms = cell( size( operators_born ) );
            end

            % ensure cell array for linear_transforms
            if ~iscell( linear_transforms )
                linear_transforms = { linear_transforms };
            end

            % overwrite properties of momentary scattering operator options
            if nargin >= 4
                operators_born = set_properties_momentary( operators_born, varargin{ 2:end } );
            end

            % multiple operators_born / single u_M
            if ~isscalar( operators_born ) && isscalar( u_M )
                u_M = repmat( u_M, size( operators_born ) );
            end

            % multiple operators_born / single linear_transforms
            if ~isscalar( operators_born ) && isscalar( linear_transforms )
                linear_transforms = repmat( linear_transforms, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, u_M, linear_transforms );

            %--------------------------------------------------------------
            % 2.) compute adjoint fluctuations
            %--------------------------------------------------------------
            % specify cell array for theta_hat
            theta_hat = cell( size( operators_born ) );

            % iterate scattering operators
            for index_object = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check mixed voltage signals and linear transform
                %----------------------------------------------------------
                % ensure class discretizations.signal_matrix
                if ~isa( u_M{ index_object }, 'discretizations.signal_matrix' )
                    errorStruct.message = sprintf( 'u_M{ %d } must be discretizations.signal_matrix!', index_object );
                    errorStruct.identifier = 'adjoint:NoSignalMatrix';
                    error( errorStruct );
                end

                % method adjoint_quick ensures class linear_transforms.linear_transform

                %----------------------------------------------------------
                % b) normalize mixed voltage signals
                %----------------------------------------------------------
                u_M{ index_object } = return_vector( u_M{ index_object } );
                u_M_norm = norm( u_M{ index_object } );
                u_M_normed = u_M{ index_object } / u_M_norm;

                %----------------------------------------------------------
                % c) quick adjoint scattering
                %----------------------------------------------------------
%                 profile on
                theta_hat{ index_object } = adjoint_quick( operators_born( index_object ), u_M_normed, linear_transforms{ index_object } );
%                 profile viewer

                % estimate samples
%                 samples_est = op_A_bar( theta_hat_weighting, 1 );

                % compute resulting error in the measurement vector
%                 y_m_res = samples( : ) - samples_est;
%                 y_m_res_l2_norm_rel = norm( y_m_res(:), 2 ) / norm( samples( : ) );

            end % for index_object = 1:numel( operators_born )

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                theta_hat = theta_hat{ 1 };
            end

        end % function theta_hat = adjoint( operators_born, u_M, varargin )

        %------------------------------------------------------------------
        % transform point spread function (overload tpsf method)
        %------------------------------------------------------------------
        function [ theta_tpsf, E_M, adjointness ] = tpsf( operators_born, indices, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'tpsf:NoOperatorsBorn';
                error( errorStruct );
            end

            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            % ensure nonempty linear_transforms
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                linear_transforms = varargin{ 1 };
            else
                % empty linear_transform is identity
                linear_transforms = cell( size( operators_born ) );
            end

            % ensure cell array for linear_transforms
            if ~iscell( linear_transforms )
                linear_transforms = { linear_transforms };
            end

            % overwrite properties of momentary scattering operator options
            if nargin >= 4
                operators_born = set_properties_momentary( operators_born, varargin{ 2:end } );
            end

            % multiple operators_born / single indices
            if ~isscalar( operators_born ) && isscalar( indices )
                indices = repmat( indices, size( operators_born ) );
            end

            % multiple operators_born / single linear_transforms
            if ~isscalar( operators_born ) && isscalar( linear_transforms )
                linear_transforms = repmat( linear_transforms, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, indices, linear_transforms );

            %--------------------------------------------------------------
            % 2.) compute TPSFs
            %--------------------------------------------------------------
            % specify cell array for tpsf
            theta_tpsf = cell( size( operators_born ) );
            E_M = cell( size( operators_born ) );
            adjointness = cell( size( operators_born ) );

            % iterate scattering operators
            for index_object = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check linear transform and indices
                %----------------------------------------------------------
                % methods forward_quick and adjoint_quick ensure class linear_transforms.linear_transform
%                 if ~isa( linear_transforms{ index_object }, 'linear_transforms.linear_transform' )
%                     errorStruct.message = sprintf( 'linear_transforms{ %d } must be linear_transforms.linear_transform!', index_object );
%                     errorStruct.identifier = 'tpsf:NoLinearTransform';
%                     error( errorStruct );
%                 end

                % numbers of TPSFs and grid points
                N_tpsf = numel( indices{ index_object } );
                N_points = operators_born( index_object ).discretization.spatial.grid_FOV.N_points;

                % ensure valid indices{ index_object }
                mustBeInteger( indices{ index_object } );
                mustBePositive( indices{ index_object } );
                mustBeLessThanOrEqual( indices{ index_object }, N_points );

                %----------------------------------------------------------
                % b) create coefficient vectors
                %----------------------------------------------------------
                % indices of coefficients
                indices_tpsf = ( 0:( N_tpsf - 1 ) ) * N_points + indices{ index_object }( : )';

                % initialize coefficient vectors
                theta = zeros( N_points, N_tpsf );
                theta( indices_tpsf ) = 1;

                %----------------------------------------------------------
                % c) quick forward scattering and received energies
                %----------------------------------------------------------
                u_M = forward_quick( operators_born( index_object ), theta, linear_transforms{ index_object } );
                E_M{ index_object } = vecnorm( u_M, 2, 1 ).^2;

                %----------------------------------------------------------
                % d) quick adjoint scattering and test for adjointness
                %----------------------------------------------------------
                theta_tpsf{ index_object } = adjoint_quick( operators_born( index_object ), u_M, linear_transforms{ index_object } );
                adjointness{ index_object } = E_M{ index_object } - theta_tpsf{ index_object }( indices_tpsf );

            end % for index_object = 1:numel( operators_born )

% TODO: create image objects

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                theta_tpsf = theta_tpsf{ 1 };
                E_M = E_M{ 1 };
                adjointness = adjointness{ 1 };
            end

        end % function [ theta_tpsf, E_M, adjointness ] = tpsf( operators_born, indices, varargin )

        %------------------------------------------------------------------
        % received energy (overload energy_rx method)
        %------------------------------------------------------------------
        function [ E_M, E_M_aa ] = energy_rx( operators_born )

% TODO: vectorize

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing received energies (CPU, Born approximation, double precision, kappa)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'energy_rx:NoOperatorsBorn';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute received energies
            %--------------------------------------------------------------
            % initialize received energies with zeros
            E_M = zeros( 1, operators_born.discretization.spatial.grid_FOV.N_points );
            E_M_aa = zeros( 1, operators_born.discretization.spatial.grid_FOV.N_points );

            % iterate sequential pulse-echo measurements
            for index_measurement = 1:numel( operators_born.discretization.spectral )

                %----------------------------------------------------------
                % prefactors
                %----------------------------------------------------------
                % map unique frequencies of pulse-echo measurement to global unique frequencies
                indices_f_measurement_to_global = operators_born.discretization.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                indices_f_mix_to_measurement = operators_born.discretization.spectral( index_measurement ).indices_f_to_unique;

                % extract prefactors for all mixes (current frequencies)
                prefactors = operators_born.discretization.prefactors{ index_measurement };

                % numbers of frequencies in mixed voltage signals
                axes_f = reshape( [ prefactors.axis ], size( prefactors ) );
                N_samples_f = abs( axes_f );

                %----------------------------------------------------------
                % compute mixed voltage signals for the active array elements
                %----------------------------------------------------------
                % iterate mixed voltage signals
                for index_mix = 1:numel( operators_born.discretization.spectral( index_measurement ).rx )

                    %------------------------------------------------------
                    % active array elements and pressure field (current frequencies)
                    %------------------------------------------------------
                    % number of active array elements
                    N_elements_active = numel( operators_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    p_incident_act = double( operators_born.incident_waves( index_measurement ).p_incident.samples( indices_f_mix_to_measurement{ index_mix }, : ) );

                    %------------------------------------------------------
                    % compute voltage signals received by the active array elements
                    %------------------------------------------------------
                    % initialize voltages with zeros
                    Phi_M = zeros( N_samples_f( index_mix ), operators_born.discretization.spatial.grid_FOV.N_points );
                    Phi_M_aa = zeros( N_samples_f( index_mix ), operators_born.discretization.spatial.grid_FOV.N_points );

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % index of active array element
                        index_element = operators_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operators_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operators_born.discretization.spatial.indices_grid_FOV_shift( :, index_element );

                            % extract current frequencies from unique frequencies
                            if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operators_born.discretization.h_ref.axis )

                                h_rx = double( operators_born.discretization.h_ref.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                                h_rx_aa = double( operators_born.h_ref_aa.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );

                            else

                                h_rx = double( operators_born.discretization.h_ref.samples( :, indices_occupied_act ) );
                                h_rx_aa = double( operators_born.h_ref_aa.samples( :, indices_occupied_act ) );
                            end

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % compute spatial transfer function of the active array element
                            h_rx = discretizations.spatial_transfer_function( operators_born.discretization.spatial, axes_f( index_mix ), index_element );
                            h_rx = double( h_rx.samples );

                        end % if isa( operators_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % compute mixed voltage signals
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        Phi_act_aa = h_rx_aa .* p_incident_act .* double( prefactors( index_mix ).samples( :, index_active ) );

                        Phi_M = Phi_M + Phi_act;
                        Phi_M_aa = Phi_M_aa + Phi_act_aa;

                    end % for index_active = 1:N_elements_active

                    %------------------------------------------------------
                    % compute energies of mixed voltage signals
                    %------------------------------------------------------
                    E_M = E_M + vecnorm( Phi_M, 2, 1 ).^2;
                    E_M_aa = E_M_aa + vecnorm( Phi_M_aa, 2, 1 ).^2;

                end % for index_mix = 1:numel( operators_born.discretization.spectral( index_measurement ).rx )

            end % for index_measurement = 1:numel( operators_born.discretization.spectral )

            %----------------------------------------------------------
            % restore physical units
            %----------------------------------------------------------
            E_M = physical_values.volt( E_M.' ) * physical_values.volt;
            E_M_aa = physical_values.volt( E_M_aa.' ) * physical_values.volt;

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function [ E_M, E_M_aa ] = energy_rx( operators_born )

        %------------------------------------------------------------------
        % matrix multiplication (overload mtimes method)
        %------------------------------------------------------------------
        function u_M = mtimes( operator_born, fluctuations )

            %--------------------------------------------------------------
            % 1.) call forward scattering
            %--------------------------------------------------------------
            u_M = forward_quick( operator_born, fluctuations );

        end % function u_M = mtimes( operator_born, fluctuations )

	end % methods

end % classdef operator_born < scattering.operator
