%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-07-31
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
            if isa( operator_born.options.momentary.gpu, 'scattering.options_gpu_off' )
                u_M = forward_quick_cpu( operator_born, fluctuations );
            else
% TODO: remove complex
                u_M = scattering.combined_quick_gpu( operator_born, 1, complex( fluctuations ) );
%                 clear mex;
            end

        end % function u_M = forward_quick( operator_born, fluctuations, varargin )

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
            if isa( operator_born.options.momentary.gpu, 'scattering.options_gpu_off' )
                gamma_hat = adjoint_quick_cpu( operator_born, u_M );
            else
                gamma_hat = scattering.combined_quick_gpu( operator_born, 2, u_M );
%                 clear mex;
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
                N_observations = { operators_born( index_object ).discretization.spectral.N_observations };
                u_M{ index_object } = mat2cell( u_M{ index_object }, cellfun( @( x ) sum( x( : ) ), N_observations( operators_born( index_object ).indices_measurement_sel ) ), size( u_M{ index_object }, 2 ) );

                % iterate selected sequential pulse-echo measurements
                for index_measurement_sel = 1:numel( operators_born( index_object ).indices_measurement_sel )

                    % index of sequential pulse-echo measurement
                    index_measurement = operator_born.indices_measurement_sel( index_measurement_sel );

                    % map unique frequencies of pulse-echo measurement to global unique frequencies
                    indices_f_measurement_to_global = operators_born( index_object ).discretization.indices_f_to_unique{ index_measurement };

                    % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                    indices_f_mix_to_measurement = operators_born( index_object ).discretization.spectral( index_measurement ).indices_f_to_unique;

                    % partition matrix into cell arrays
                    u_M{ index_object }{ index_measurement_sel } = mat2cell( u_M{ index_object }{ index_measurement_sel }, operators_born( index_object ).discretization.spectral( index_measurement ).N_observations, size( u_M{ index_object }{ index_measurement_sel }, 2 ) );

                    % subsample global unique frequencies to get unique frequencies of pulse-echo measurement
                    axis_f_measurement_unique = subsample( operators_born( index_object ).discretization.axis_f_unique, indices_f_measurement_to_global );

                    % subsample unique frequencies of pulse-echo measurement to get frequencies of mixed voltage signals
                    axes_f_mix = reshape( subsample( axis_f_measurement_unique, indices_f_mix_to_measurement ), size( u_M{ index_object }{ index_measurement_sel } ) );

                    % create mixed voltage signals
                    u_M{ index_object }{ index_measurement_sel } = discretizations.signal( axes_f_mix, u_M{ index_object }{ index_measurement_sel } );

                    % try to merge mixed voltage signals
                    try
                        u_M{ index_object }{ index_measurement_sel } = merge( u_M{ index_object }{ index_measurement_sel } );
                    catch
                    end

                end % for index_measurement_sel = 1:numel( operators_born( index_object ).indices_measurement_sel )

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

%             discretizations.signal_matrix()
% TODO: create field objects

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                theta_tpsf = theta_tpsf{ 1 };
                E_M = E_M{ 1 };
                adjointness = adjointness{ 1 };
            end

        end % function [ theta_tpsf, E_M, adjointness ] = tpsf( operators_born, indices, varargin )

        %------------------------------------------------------------------
        % received energy (wrapper)
        %------------------------------------------------------------------
        function E_M = energy_rx( operators_born, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'energy_rx:NoOperatorsBorn';
                error( errorStruct );
            end

            % ensure nonempty linear_transforms
            if nargin >= 2 && ~isempty( varargin{ 1 } )
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
            if nargin >= 3
                operators_born = set_properties_momentary( operators_born, varargin{ 2:end } );
            end

            % multiple operators_born / single linear_transforms
            if ~isscalar( operators_born ) && isscalar( linear_transforms )
                linear_transforms = repmat( linear_transforms, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, linear_transforms );

            %--------------------------------------------------------------
            % 2.) compute received energies
            %--------------------------------------------------------------
            % specify cell array for E_M
            E_M = cell( size( operators_born ) );

            % iterate scattering operators
            for index_object = 1:numel( operators_born )

                % extract indices of selected sequential pulse-echo measurements
                indices_measurement_sel = operators_born( index_object ).indices_measurement_sel;

                % check linear transform
                if ~isempty( linear_transforms{ index_object } )

                    %------------------------------------------------------
                    % a) arbitrary linear transform
                    %------------------------------------------------------
                    % initialize results w/ zeros
                    E_M{ index_object } = physical_values.squarevolt( zeros( linear_transforms{ index_object }.N_coefficients, numel( operators_born( index_object ).indices_measurement_sel ) ) );

                    % iterate selected sequential pulse-echo measurements
                    for index_measurement_sel = 1:numel( indices_measurement_sel )

                        % index of sequential pulse-echo measurement
                        index_measurement = indices_measurement_sel( index_measurement_sel );

                        % set momentary scattering operator options
                        operators_born( index_object ) = set_properties_momentary( operators_born( index_object ), scattering.options_sequence_selected( index_measurement ) );

                        % create format string for filename
                        str_format = sprintf( 'data/%s/spatial_%%s/E_M_spectral_%%s_transform_%%s_options_aliasing_%%s.mat', operators_born( index_object ).discretization.spatial.str_name );

                        % load or compute received energies (arbitrary linear transform)
                        E_M{ index_object }( :, index_measurement_sel ) = auxiliary.compute_or_load_hash( str_format, @energy_rx_arbitrary, [ 2, 3, 4, 5 ], [ 1, 4 ], operators_born( index_object ), operators_born( index_object ).discretization.spatial, operators_born( index_object ).discretization.spectral( index_measurement ), linear_transforms{ index_object }, operators_born( index_object ).options.momentary.anti_aliasing );

                    end % for index_measurement_sel = 1:numel( indices_measurement_sel )

                else

                    %------------------------------------------------------
                    % b) canonical basis
                    %------------------------------------------------------
                    % initialize results w/ zeros
                    E_M{ index_object } = physical_values.squarevolt( zeros( operators_born( index_object ).discretization.spatial.grid_FOV.N_points, numel( operators_born( index_object ).indices_measurement_sel ) ) );

                    % iterate selected sequential pulse-echo measurements
                    for index_measurement_sel = 1:numel( indices_measurement_sel )

                        % index of sequential pulse-echo measurement
                        index_measurement = indices_measurement_sel( index_measurement_sel );

                        % set momentary scattering operator options
                        operators_born( index_object ) = set_properties_momentary( operators_born( index_object ), scattering.options_sequence_selected( index_measurement ) );

                        % create format string for filename
                        str_format = sprintf( 'data/%s/spatial_%%s/E_M_spectral_%%s_options_aliasing_%%s.mat', operators_born( index_object ).discretization.spatial.str_name );

                        % load or compute received energies (canonical basis)
                        E_M{ index_object }( :, index_measurement_sel ) = auxiliary.compute_or_load_hash( str_format, @energy_rx_canonical, [ 2, 3, 4 ], 1, operators_born( index_object ), operators_born( index_object ).discretization.spatial, operators_born( index_object ).discretization.spectral( index_measurement ), operators_born( index_object ).options.momentary.anti_aliasing );

                    end % for index_measurement_sel = 1:numel( indices_measurement_sel )

                end % if ~isempty( linear_transforms{ index_object } )

                % sum received energies
                E_M{ index_object } = sum( E_M{ index_object }, 2 );

            end % for index_object = 1:numel( operators_born )

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                E_M = E_M{ 1 };
            end

        end % function E_M = energy_rx( operators_born, varargin )

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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

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
            u_M = cell( numel( operator_born.indices_measurement_sel ), 1 );

            % iterate selected sequential pulse-echo measurements
            for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

                % index of sequential pulse-echo measurement
                index_measurement = operator_born.indices_measurement_sel( index_measurement_sel );

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
                % specify cell arrays for u_M{ index_measurement_sel }
                u_M{ index_measurement_sel } = cell( size( operator_born.discretization.spectral( index_measurement ).rx ) );

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
                    u_M{ index_measurement_sel }{ index_mix } = zeros( N_samples_f( index_mix ), N_objects );

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
                            if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.h_ref_aa.axis )
                                h_rx = double( operator_born.h_ref_aa.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                            else
                                h_rx = double( operator_born.h_ref_aa.samples( :, indices_occupied_act ) );
                            end

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % compute spatial transfer function of the active array element
                            h_rx = transfer_function( operator_born.discretization.spatial, axes_f( index_mix ), index_element );

                            % apply spatial anti-aliasing filter
                            h_rx = discretizations.anti_aliasing_filter( operator_born.sequence.setup.xdc_array, operator_born.sequence.setup.homogeneous_fluid, h_rx, operator_born.options.momentary.anti_aliasing );
                            h_rx = double( h_rx.samples );

                        end % if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % compute matrix-vector product and mix voltage signals
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_occupied_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        u_M{ index_measurement_sel }{ index_mix } = u_M{ index_measurement_sel }{ index_mix } + Phi_act * fluctuations( indices_occupied, : );

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                % concatenate cell array contents into matrix
                u_M{ index_measurement_sel } = cat( 1, u_M{ index_measurement_sel }{ : } );

            end % for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

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
            N_obs = { operator_born.discretization.spectral.N_observations };
            u_M = mat2cell( u_M, cellfun( @( x ) sum( x( : ) ), N_obs( operator_born.indices_measurement_sel ) ), size( u_M, 2 ) );

            % iterate selected sequential pulse-echo measurements
            for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

                % index of sequential pulse-echo measurement
                index_measurement = operator_born.indices_measurement_sel( index_measurement_sel );

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
                u_M{ index_measurement_sel } = mat2cell( u_M{ index_measurement_sel }, operator_born.discretization.spectral( index_measurement ).N_observations, size( u_M{ index_measurement_sel }, 2 ) );

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
                            if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.h_ref_aa.axis )
                                h_rx = double( operator_born.h_ref_aa.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                            else
                                h_rx = double( operator_born.h_ref_aa.samples( :, indices_occupied_act ) );
                            end

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % compute spatial transfer function of the active array element
                            h_rx = transfer_function( operator_born.discretization.spatial, axes_f( index_mix ), index_element );

                            % apply spatial anti-aliasing filter
                            h_rx = discretizations.anti_aliasing_filter( operator_born.sequence.setup.xdc_array, operator_born.sequence.setup.homogeneous_fluid, h_rx, operator_born.options.momentary.anti_aliasing );
                            h_rx = double( h_rx.samples );

                        end % if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % compute matrix-vector product
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        gamma_hat = gamma_hat + Phi_act' * u_M{ index_measurement_sel }{ index_mix };

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

            end % for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function gamma_hat = adjoint_quick_cpu( operator_born, u_M )

        %------------------------------------------------------------------
        % quick adjoint scattering (GPU: C++ & CUDA API)
        %------------------------------------------------------------------
        % see combined_quick_gpu.cu

        %------------------------------------------------------------------
        % received energy (arbitrary linear transform)
        %------------------------------------------------------------------
        function E_M = energy_rx_arbitrary( operator_born, linear_transform )

            % internal constant
            N_objects = 164;

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing received energies (Born approximation, kappa, arbitrary linear transform)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born (scalar)
            if ~( isa( operator_born, 'scattering.operator_born' ) && isscalar( operator_born ) )
                errorStruct.message = 'operator_born must be a single scattering.operator_born!';
                errorStruct.identifier = 'energy_rx:NoSingleOperatorBorn';
                error( errorStruct );
            end

            % ensure class linear_transforms.linear_transform (scalar)
            if ~( isa( linear_transform, 'linear_transforms.linear_transform' ) && isscalar( linear_transform ) )
                errorStruct.message = 'linear_transform must be a single linear_transforms.linear_transform!';
                errorStruct.identifier = 'energy_rx:NoSingleLinearTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute received energies
            %--------------------------------------------------------------
            % compute number of batches and objects in last batch
            N_batches = ceil( linear_transform.N_coefficients / N_objects );
            N_objects_last = linear_transform.N_coefficients - ( N_batches - 1 ) * N_objects;

            % partition indices of transform coefficients into N_batches batches
            indices_coeff = mat2cell( ( 1:linear_transform.N_coefficients ), 1, [ N_objects * ones( 1, N_batches - 1 ), N_objects_last ] );

            % initialize received energies with zeros
            E_M = zeros( linear_transform.N_coefficients, 1 );

            % name for temporary file
            str_filename = sprintf( 'data/%s/energy_rx_temp.mat', operator_born.discretization.spatial.str_name );

            % get name of directory
            [ str_name_dir, ~, ~ ] = fileparts( str_filename );

            % ensure existence of folder str_name_dir
            [ success, errorStruct.message, errorStruct.identifier ] = mkdir( str_name_dir );
            if ~success
                error( errorStruct );
            end

            % iterate batches of transform coefficients
            for index_batch = 1%:N_batches

                % print progress in percent
                fprintf( '%5.1f %%', ( index_batch - 1 ) / N_batches * 1e2 );

                %----------------------------------------------------------
                % a) create batch of coefficient vectors
                %----------------------------------------------------------
                % indices of transform coefficients
                indices_theta = ( 0:( numel( indices_coeff{ index_batch } ) - 1 ) ) * linear_transform.N_coefficients + indices_coeff{ index_batch };

                % initialize transform coefficients
                theta_kappa = zeros( linear_transform.N_coefficients, numel( indices_coeff{ index_batch } ) );
                theta_kappa( indices_theta ) = 1;

                %----------------------------------------------------------
                % b) quick forward scattering and received energies
                %----------------------------------------------------------
                % quick forward scattering
                u_M = forward_quick( operator_born, theta_kappa, linear_transform );

                % compute received energy
                E_M( indices_coeff{ index_batch } ) = vecnorm( u_M, 2, 1 ).^2;

                %----------------------------------------------------------
                % c) save and display intermediate results
                %----------------------------------------------------------
                % save intermediate results
                save( str_filename, 'E_M' );

                % display intermediate results
                figure( 999 );
                imagesc( illustration.dB( squeeze( reshape( E_M, operator_born.discretization.spatial.grid_FOV.N_points_axis ) )', 10 ), [ -60, 0 ] );

                % erase progress in percent
                fprintf( '\b\b\b\b\b\b\b' );

            end % for index_batch = 1:N_batches

            %--------------------------------------------------------------
            % 3.) restore physical units
            %--------------------------------------------------------------
            E_M = physical_values.squarevolt( E_M );

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_M = energy_rx_arbitrary( operator_born, linear_transform )

        %------------------------------------------------------------------
        % received energy (canonical basis)
        %------------------------------------------------------------------
        function E_M = energy_rx_canonical( operator_born )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing received energies (CPU, Born approximation, double precision, kappa, canonical basis)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born (scalar)
            if ~( isa( operator_born, 'scattering.operator_born' ) && isscalar( operator_born ) )
                errorStruct.message = 'operator_born must be a single scattering.operator_born!';
                errorStruct.identifier = 'energy_rx_canonical:NoSingleOperatorsBorn';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute received energies
            %--------------------------------------------------------------
            % initialize received energies with zeros
            E_M = zeros( 1, operator_born.discretization.spatial.grid_FOV.N_points );

            % iterate selected sequential pulse-echo measurements
            for index_measurement = operator_born.indices_measurement_sel

                %----------------------------------------------------------
                % a) prefactors
                %----------------------------------------------------------
                % map unique frequencies of pulse-echo measurement to global unique frequencies
                indices_f_measurement_to_global = operator_born.discretization.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                indices_f_mix_to_measurement = operator_born.discretization.spectral( index_measurement ).indices_f_to_unique;

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.discretization.prefactors{ index_measurement };

                % numbers of frequencies in mixed voltage signals
                axes_f = reshape( [ prefactors.axis ], size( prefactors ) );
                N_samples_f = abs( axes_f );

                %----------------------------------------------------------
                % b) compute mixed voltage signals for the active array elements
                %----------------------------------------------------------
                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

                    %------------------------------------------------------
                    % active array elements and pressure field (current frequencies)
                    %------------------------------------------------------
                    % number of active array elements
                    N_elements_active = numel( operator_born.discretization.spectral( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    p_incident_act = double( operator_born.incident_waves( index_measurement ).p_incident.samples( indices_f_mix_to_measurement{ index_mix }, : ) );

                    %------------------------------------------------------
                    % compute voltage signals received by the active array elements
                    %------------------------------------------------------
                    % initialize voltages with zeros
                    Phi_M = zeros( N_samples_f( index_mix ), operator_born.discretization.spatial.grid_FOV.N_points );

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
                            if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.h_ref_aa.axis )
                                h_rx = double( operator_born.h_ref_aa.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                            else
                                h_rx = double( operator_born.h_ref_aa.samples( :, indices_occupied_act ) );
                            end

                        else

                            %----------------------------------------------
                            % b) arbitrary grid
                            %----------------------------------------------
                            % compute spatial transfer function of the active array element
                            h_rx = transfer_function( operator_born.discretization.spatial, axes_f( index_mix ), index_element );

                            % apply spatial anti-aliasing filter
                            h_rx = discretizations.anti_aliasing_filter( operator_born.sequence.setup.xdc_array, operator_born.sequence.setup.homogeneous_fluid, h_rx, operator_born.options.momentary.anti_aliasing );
                            h_rx = double( h_rx.samples );

                        end % if isa( operator_born.discretization.spatial, 'discretizations.spatial_grid_symmetric' )

                        %--------------------------------------------------
                        % compute mixed voltage signals
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        Phi_M = Phi_M + Phi_act;

                    end % for index_active = 1:N_elements_active

                    %------------------------------------------------------
                    % compute energies of mixed voltage signals
                    %------------------------------------------------------
                    E_M = E_M + vecnorm( Phi_M, 2, 1 ).^2;

                end % for index_mix = 1:numel( operator_born.discretization.spectral( index_measurement ).rx )

            end % for index_measurement = operator_born.indices_measurement_sel

            %--------------------------------------------------------------
            % 3.) restore physical units
            %--------------------------------------------------------------
            E_M = physical_values.squarevolt( E_M.' );

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_M = energy_rx_canonical( operator_born )

    end % methods (Access = private, Hidden)

end % classdef operator_born < scattering.operator
