%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2020-02-18
%
classdef operator_born < scattering.operator

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = operator_born( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class scattering.sequences.sequence for sequences
            % superclass ensures class scattering.options for options

            %--------------------------------------------------------------
            % 2.) create scattering operators based on the Born approximation
            %--------------------------------------------------------------
            % constructor of superclass
            object@scattering.operator( sequences, options );

        end % function object = operator_born( sequences, options )

        %------------------------------------------------------------------
        % quick forward scattering (wrapper)
        %------------------------------------------------------------------
        function u_M = forward_quick( operator_born, coefficients, varargin )

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
            if ~( isnumeric( coefficients ) && ismatrix( coefficients ) )
                errorStruct.message = 'coefficients must be a numeric matrix!';
                errorStruct.identifier = 'forward_quick:NoNumericMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) dictionary synthesis
            %--------------------------------------------------------------
            if nargin >= 3 && ~isempty( varargin{ 1 } )

                % ensure class linear_transforms.linear_transform (scalar)
                if ~( isa( varargin{ 1 }, 'linear_transforms.linear_transform' ) && isscalar( varargin{ 1 } ) )
                    errorStruct.message = 'Nonempty varargin{ 1 } must be a single linear_transforms.linear_transform!';
                    errorStruct.identifier = 'forward_quick:NoSingleLinearTransform';
                    error( errorStruct );
                end

                % apply adjoint linear transform
                coefficients = adjoint_transform( varargin{ 1 }, coefficients );

            end % if nargin >= 3 && ~isempty( varargin{ 1 } )

            %--------------------------------------------------------------
            % 3.) compute mixed voltage signals
            %--------------------------------------------------------------
            if isa( operator_born.options.momentary.gpu, 'scattering.options.gpu_off' )
                u_M = forward_quick_cpu( operator_born, coefficients );
            else
                u_M = scattering.combined_quick_gpu( operator_born, 1, coefficients );
%                 clear mex;
            end

            %--------------------------------------------------------------
            % 4.) time gain compensation (TGC)
            %--------------------------------------------------------------
            if nargin >= 4 && ~isempty( varargin{ 2 } )

                % ensure class linear_transforms.linear_transform (scalar)
                if ~( isa( varargin{ 2 }, 'linear_transforms.linear_transform' ) && isscalar( varargin{ 2 } ) )
                    errorStruct.message = 'Nonempty varargin{ 2 } must be a single linear_transforms.linear_transform!';
                    errorStruct.identifier = 'forward_quick:NoSingleLinearTransform';
                    error( errorStruct );
                end

                % apply time gain compensation
                u_M = forward_transform( varargin{ 2 }, u_M );

            end % if nargin >= 4 && ~isempty( varargin{ 2 } )

        end % function u_M = forward_quick( operator_born, coefficients, varargin )

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
            % 2.) adjoint time gain compensation (TGC)
            %--------------------------------------------------------------
            if nargin >= 4 && ~isempty( varargin{ 2 } )

                % ensure class linear_transforms.linear_transform (scalar)
                if ~( isa( varargin{ 2 }, 'linear_transforms.linear_transform' ) && isscalar( varargin{ 2 } ) )
                    errorStruct.message = 'Nonempty varargin{ 2 } must be a single linear_transforms.linear_transform!';
                    errorStruct.identifier = 'adjoint_quick:NoSingleLinearTransform';
                    error( errorStruct );
                end

                % apply adjoint time gain compensation
                u_M = adjoint_transform( varargin{ 2 }, u_M );

            end % if nargin >= 4 && ~isempty( varargin{ 2 } )

            %--------------------------------------------------------------
            % 3.) compute adjoint fluctuations
            %--------------------------------------------------------------
            if isa( operator_born.options.momentary.gpu, 'scattering.options.gpu_off' )
                theta_hat = adjoint_quick_cpu( operator_born, u_M );
            else
                theta_hat = scattering.combined_quick_gpu( operator_born, 2, u_M );
%                 clear mex;
            end

            temp_1 = squeeze( reshape( abs( theta_hat( :, 1 ) ), operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) );

            %--------------------------------------------------------------
            % 4.) dictionary analysis
            %--------------------------------------------------------------
            if nargin >= 3 && ~isempty( varargin{ 1 } )

                % ensure class linear_transforms.linear_transform (scalar)
                if ~( isa( varargin{ 1 }, 'linear_transforms.linear_transform' ) && isscalar( varargin{ 1 } ) )
                    errorStruct.message = 'Nonempty varargin{ 1 } must be a single linear_transforms.linear_transform!';
                    errorStruct.identifier = 'adjoint_quick:NoSingleLinearTransform';
                    error( errorStruct );
                end

                % apply forward linear transform
                theta_hat = forward_transform( varargin{ 1 }, theta_hat );

            end % if nargin >= 3 && ~isempty( varargin{ 1 } )

            % illustrate
%             temp_2 = squeeze( reshape( abs( theta_hat( :, 1 ) ), operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) );
            figure(999);
            if ismatrix( temp_1 )
                subplot( 1, 2, 1 );
                imagesc( illustration.dB( temp_1, 20 )', [ -60, 0 ] );
                subplot( 1, 2, 2 );
%                 imagesc( illustration.dB( temp_2, 20 )', [ -60, 0 ] );
            else
                subplot( 1, 2, 1 );
                imagesc( illustration.dB( squeeze( temp_1( :, 5, : ) ), 20 )', [ -60, 0 ] );
                subplot( 1, 2, 2 );
%                 imagesc( illustration.dB( squeeze( temp_2( :, 5, : ) ), 20 )', [ -60, 0 ] );
            end

        end % function theta_hat = adjoint_quick( operator_born, u_M, varargin )

        %------------------------------------------------------------------
        % quick combined scattering
        %------------------------------------------------------------------
        function y = combined_quick( operator_born, mode, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born (scalar)
            if ~( isa( operator_born, 'scattering.operator_born' ) && isscalar( operator_born ) )
                errorStruct.message = 'operator_born must be a single scattering.operator_born!';
                errorStruct.identifier = 'adjoint_quick:NoSingleOperatorBorn';
                error( errorStruct );
            end

            % ensure nonempty nonnegative integer for mode
            mustBeNonnegative( mode );
            mustBeInteger( mode );

            % functions forward_quick or adjoint_quick ensure numeric matrix for x

            %--------------------------------------------------------------
            % 2.) quick combined scattering
            %--------------------------------------------------------------
            switch mode

                case 0

                    %------------------------------------------------------
                    % a) return size of forward transform
                    %------------------------------------------------------
                    N_observations = cellfun( @( x ) sum( x( : ) ), { operator_born.sequence.settings( operator_born.indices_measurement_sel ).N_observations } );
                    N_observations = sum( N_observations( : ) );
% TODO: wrong! number of coefficients in 2nd entry
                    y = [ N_observations, operator_born.sequence.size( 2 ) ];

                case 1

                    %------------------------------------------------------
                    % b) quick forward scattering (wrapper)
                    %------------------------------------------------------
                    y = forward_quick( operator_born, varargin{ : } );

                case 2

                    %------------------------------------------------------
                    % c) quick adjoint scattering (wrapper)
                    %------------------------------------------------------
                    y = adjoint_quick( operator_born, varargin{ : } );

                otherwise

                    %------------------------------------------------------
                    % d) unknown operation
                    %------------------------------------------------------
                    errorStruct.message = 'Unknown mode of operation!';
                    errorStruct.identifier = 'combined_quick:InvalidMode';
                    error( errorStruct );

            end % switch mode

        end % function y = combined_quick( operator_born, mode, varargin )

        %------------------------------------------------------------------
        % forward scattering (overload forward method)
        %------------------------------------------------------------------
        function u_M = forward( operators_born, fluctuations, options )

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

            % ensure nonempty options
            if nargin <= 2 || isempty( options )
                options = regularization.options.common;
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % multiple operators_born / single options
            if ~isscalar( operators_born ) && isscalar( options )
                options = repmat( options, size( operators_born ) );
            end

            % single operators_born / multiple options
            if isscalar( operators_born ) && ~isscalar( options )
                operators_born = repmat( operators_born, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, fluctuations, options );

            %--------------------------------------------------------------
            % 2.) compute mixed voltage signals
            %--------------------------------------------------------------
            % specify cell array for u_M
            u_M = cell( size( operators_born ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class regularization.options.common
                if ~isa( options{ index_operator }, 'regularization.options.common' )
                    errorStruct.message = sprintf( 'options{ %d } must be regularization.options.common!', index_operator );
                    errorStruct.identifier = 'forward:NoCommonOptions';
                    error( errorStruct );
                end

                % ensure numeric matrix
                if ~( isnumeric( fluctuations{ index_operator } ) && ismatrix( fluctuations{ index_operator } ) )
                    errorStruct.message = sprintf( 'fluctuations{ %d } must be a numeric matrix!', index_operator );
                    errorStruct.identifier = 'forward:NoNumericMatrix';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % specify cell array for u_M{ index_operator }
                u_M{ index_operator } = cell( size( options{ index_operator } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator } )

                    %------------------------------------------------------
                    % i.) create configuration
                    %------------------------------------------------------
                    [ operator_born_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators_born( index_operator ) );

                    %------------------------------------------------------
                    % ii.) quick forward scattering
                    %------------------------------------------------------
                    u_M{ index_operator }{ index_options } = forward_quick( operator_born_act, fluctuations{ index_operator }, LT_dict_act, LT_tgc_act );
                    u_M{ index_operator }{ index_options } = physical_values.volt( u_M{ index_operator }{ index_options } );

                    % create signals or signal matrices
                    u_M{ index_operator }{ index_options } = format_voltages( operator_born_act, u_M{ index_operator }{ index_options } );

                end % for index_options = 1:numel( options{ index_operator } )

                % avoid cell array for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    u_M{ index_operator } = u_M{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators_born )

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                u_M = u_M{ 1 };
            end

        end % function u_M = forward( operators_born, fluctuations, options )

        %------------------------------------------------------------------
        % adjoint scattering (overload adjoint method)
        %------------------------------------------------------------------
        function [ gamma_hat, theta_hat, rel_RMSE ] = adjoint( operators_born, u_M, options )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: adjoint scattering...\n', str_date_time );

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

            % ensure nonempty options
            if nargin < 3 || isempty( options )
                options = regularization.options.common;
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % method get_configs ensures class regularization.options.common for options

            % multiple operators_born / single options
            if ~isscalar( operators_born ) && isscalar( options )
                options = repmat( options, size( operators_born ) );
            end

            % single operators_born / multiple options
            if isscalar( operators_born ) && ~isscalar( options )
                operators_born = repmat( operators_born, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, u_M, options );

            %--------------------------------------------------------------
            % 2.) compute adjoint scattering
            %--------------------------------------------------------------
            % specify cell arrays
            gamma_hat = cell( size( operators_born ) );
            theta_hat = cell( size( operators_born ) );
            rel_RMSE = cell( size( operators_born ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class processing.signal_matrix
                if ~isa( u_M{ index_operator }, 'processing.signal_matrix' )
                    errorStruct.message = sprintf( 'u_M{ %d } must be processing.signal_matrix!', index_operator );
                    errorStruct.identifier = 'adjoint:NoSignalMatrices';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % specify cell arrays
                gamma_hat{ index_operator } = cell( size( options{ index_operator } ) );
                theta_hat{ index_operator } = cell( size( options{ index_operator } ) );
                rel_RMSE{ index_operator } = cell( size( options{ index_operator } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator } )

                    % display options
                    show( options{ index_operator }( index_options ) );

                    %------------------------------------------------------
                    % i.) create configuration
                    %------------------------------------------------------
                    [ operator_born_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators_born( index_operator ) );

                    %------------------------------------------------------
                    % ii.) create mixed voltage signals
                    %------------------------------------------------------
                    % extract relevant mixed voltage signals
% TODO: detect configuration changes first and avoid step if necessary
                    u_M_act = u_M{ index_operator }( operator_born_act.indices_measurement_sel );

                    % apply TGC and normalize mixed voltage signals
                    u_M_act_vect = return_vector( u_M_act );
                    u_M_act_vect_tgc = forward_transform( LT_tgc_act, u_M_act_vect );
                    u_M_act_vect_tgc_norm = norm( u_M_act_vect_tgc );
                    u_M_act_vect_tgc_normed = u_M_act_vect_tgc / u_M_act_vect_tgc_norm;

                    %------------------------------------------------------
                    % iii.) quick adjoint scattering
                    %------------------------------------------------------
                    theta_hat{ index_operator }{ index_options } = adjoint_quick( operator_born_act, u_M_act_vect_tgc_normed, LT_dict_act, LT_tgc_act );

                    %------------------------------------------------------
                    % iv.) apply adjoint linear transform
                    %------------------------------------------------------
                    gamma_hat{ index_operator }{ index_options } = adjoint_transform( LT_dict_act, theta_hat{ index_operator }{ index_options } );

                    %------------------------------------------------------
                    % v.) relative RMSEs by quick forward scattering
                    %------------------------------------------------------
                    if nargout >= 3

                        % estimate normalized mixed voltage signals by quick forward scattering
                        u_M_act_vect_tgc_normed_est = forward_quick( operator_born_act, theta_hat{ index_operator }{ index_options }, LT_dict_act, LT_tgc_act );

                        % compute relative RMSE
                        u_M_act_vect_tgc_normed_res = u_M_act_vect_tgc_normed - u_M_act_vect_tgc_normed_est;
                        rel_RMSE{ index_operator }{ index_options } = norm( u_M_act_vect_tgc_normed_res( : ), 2 );

                    end % if nargout >= 3

                end % for index_options = 1:numel( options{ index_operator } )

                %----------------------------------------------------------
                % c) create images
                %----------------------------------------------------------
                gamma_hat{ index_operator } = processing.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, gamma_hat{ index_operator } );

                % avoid cell arrays for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    theta_hat{ index_operator } = theta_hat{ index_operator }{ 1 };
                    rel_RMSE{ index_operator } = rel_RMSE{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators_born )

            % avoid cell arrays for single operators_born
            if isscalar( operators_born )
                gamma_hat = gamma_hat{ 1 };
                theta_hat = theta_hat{ 1 };
                rel_RMSE = rel_RMSE{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function [ gamma_hat, theta_hat, rel_RMSE ] = adjoint( operators_born, u_M, options )

        %------------------------------------------------------------------
        % transform point spread function (overload tpsf method)
        %------------------------------------------------------------------
        function [ theta_tpsf, E_M, adjointness ] = tpsf( operators_born, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'tpsf:NoOperatorsBorn';
                error( errorStruct );
            end

            % ensure nonempty options
            if nargin <= 1 || isempty( options )
                options = regularization.options.tpsf;
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % multiple operators_born / single options
            if ~isscalar( operators_born ) && isscalar( options )
                options = repmat( options, size( operators_born ) );
            end

            % single operators_born / multiple options
            if isscalar( operators_born ) && ~isscalar( options )
                operators_born = repmat( operators_born, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, options );

            %--------------------------------------------------------------
            % 2.) compute transform point spread functions (TPSFs)
            %--------------------------------------------------------------
            % specify cell arrays
            theta_tpsf = cell( size( operators_born ) );
            E_M = cell( size( operators_born ) );
            adjointness = cell( size( operators_born ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class regularization.options.tpsf
                if ~isa( options{ index_operator }, 'regularization.options.tpsf' )
                    errorStruct.message = sprintf( 'options{ %d } must be regularization.options.tpsf!', index_operator );
                    errorStruct.identifier = 'adjoint:NoOptionsTPSF';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % specify cell arrays
                theta_tpsf{ index_operator } = cell( size( options{ index_operator } ) );
                E_M{ index_operator } = cell( size( options{ index_operator } ) );
                adjointness{ index_operator } = cell( size( options{ index_operator } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator } )

                    %------------------------------------------------------
                    % i.) create configuration
                    %------------------------------------------------------
                    [ operator_born_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators_born( index_operator ) );

                    %------------------------------------------------------
                    % ii.) create coefficient vectors
                    %------------------------------------------------------
                    % a) number of TPSFs
                    N_tpsf = numel( options{ index_operator }( index_options ).indices );

                    % b) indices of coefficients
                    indices_tpsf = ( 0:( N_tpsf - 1 ) ) * LT_dict_act.N_coefficients + options{ index_operator }( index_options ).indices';

                    % c) initialize coefficient vectors
                    theta = zeros( LT_dict_act.N_coefficients, N_tpsf );
                    theta( indices_tpsf ) = 1;

                    %------------------------------------------------------
                    % iii.) quick forward scattering and received energies
                    %------------------------------------------------------
                    u_M = forward_quick( operator_born_act, theta, LT_dict_act, LT_tgc_act );
                    E_M{ index_operator }{ index_options } = vecnorm( u_M, 2, 1 ).^2;

                    %------------------------------------------------------
                    % iv.) quick adjoint scattering and test for adjointness
                    %------------------------------------------------------
                    theta_tpsf{ index_operator }{ index_options } = adjoint_quick( operator_born_act, u_M, LT_dict_act, LT_tgc_act );
                    adjointness{ index_operator }{ index_options } = E_M{ index_operator }{ index_options } - theta_tpsf{ index_operator }{ index_options }( indices_tpsf );

                end % for index_options = 1:numel( options{ index_operator } )

                % avoid cell array for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    E_M{ index_operator } = E_M{ index_operator }{ 1 };
                    adjointness{ index_operator } = adjointness{ index_operator }{ 1 };
                end

                %----------------------------------------------------------
                % c) create images
                %----------------------------------------------------------
                theta_tpsf{ index_operator } ...
                = processing.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, ...
                                         theta_tpsf{ index_operator } );

            end % for index_operator = 1:numel( operators_born )

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                theta_tpsf = theta_tpsf{ 1 };
                E_M = E_M{ 1 };
                adjointness = adjointness{ 1 };
            end

        end % function [ theta_tpsf, E_M, adjointness ] = tpsf( operators_born, options )

        %------------------------------------------------------------------
        % matrix multiplication (overload mtimes method)
        %------------------------------------------------------------------
        function u_M = mtimes( operator_born, fluctuations )

            %--------------------------------------------------------------
            % 1.) quick forward scattering
            %--------------------------------------------------------------
            u_M = forward_quick( operator_born, fluctuations );

        end % function u_M = mtimes( operator_born, fluctuations )

        %------------------------------------------------------------------
        % format voltages
        %------------------------------------------------------------------
        function u_M = format_voltages( operators_born, u_M )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'format_voltages:NoOperatorsBorn';
                error( errorStruct );
            end

            % ensure cell array for u_M
            if ~iscell( u_M )
                u_M = { u_M };
            end

            % multiple operators_born / single u_M
            if ~isscalar( operators_born ) && isscalar( u_M )
                u_M = repmat( u_M, size( operators_born ) );
            end

            % single operators_born / multiple u_M
            if isscalar( operators_born ) && ~isscalar( u_M )
                operators_born = repmat( operators_born, size( u_M ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, u_M );

            %--------------------------------------------------------------
            % 2.) format voltages
            %--------------------------------------------------------------
            % iterate scattering operators
            for index_object = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure numeric matrix
                if ~( isnumeric( u_M{ index_object } ) && ismatrix( u_M{ index_object } ) )
                    errorStruct.message = sprintf( 'u_M{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'format_voltages:NoNumericMatrix';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) create signals or signal matrices
                %----------------------------------------------------------
                % number of columns
                N_columns = size( u_M{ index_object }, 2 );

                % partition numeric matrix into cell arrays
                N_observations = cellfun( @( x ) sum( x( : ) ), { operators_born( index_object ).sequence.settings( operators_born( index_object ).indices_measurement_sel ).N_observations } );
                u_M{ index_object } = mat2cell( u_M{ index_object }, N_observations, N_columns );

                % iterate selected sequential pulse-echo measurements
                for index_measurement_sel = 1:numel( operators_born( index_object ).indices_measurement_sel )

                    % index of sequential pulse-echo measurement
                    index_measurement = operators_born( index_object ).indices_measurement_sel( index_measurement_sel );

                    % map unique frequencies of pulse-echo measurement to global unique frequencies
                    indices_f_measurement_to_global = operators_born( index_object ).sequence.indices_f_to_unique{ index_measurement };

                    % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                    indices_f_mix_to_measurement = operators_born( index_object ).sequence.settings( index_measurement ).indices_f_to_unique;

                    % partition matrix into cell arrays
                    u_M{ index_object }{ index_measurement_sel } = mat2cell( u_M{ index_object }{ index_measurement_sel }, operators_born( index_object ).sequence.settings( index_measurement ).N_observations, ones( 1, N_columns ) );

                    % subsample global unique frequencies to get unique frequencies of pulse-echo measurement
                    axis_f_measurement_unique = subsample( operators_born( index_object ).sequence.axis_f_unique, indices_f_measurement_to_global );

                    % subsample unique frequencies of pulse-echo measurement to get frequencies of mixed voltage signals
                    axes_f_mix = reshape( subsample( axis_f_measurement_unique, indices_f_mix_to_measurement ), [ size( u_M{ index_object }{ index_measurement_sel }, 1 ), 1 ] );

                    % iterate objects
                    signals = cell( 1, N_columns );
                    for index_column = 1:N_columns

                        % create mixed voltage signals
                        signals{ index_column } = processing.signal( axes_f_mix, u_M{ index_object }{ index_measurement_sel }( :, index_column ) );

                        % try to merge mixed voltage signals
                        try
                            signals{ index_column } = merge( signals{ index_column } );
                        catch
                        end

                    end % for index_column = 1:N_columns

                    % store results in cell array
                    u_M{ index_object }{ index_measurement_sel } = signals;

                    % create array of signal matrices
                    if all( cellfun( @( x ) strcmp( class( x ), 'processing.signal_matrix' ), u_M{ index_object }{ index_measurement_sel } ) )
                        u_M{ index_object }{ index_measurement_sel } = cat( 2, u_M{ index_object }{ index_measurement_sel }{ : } );
                    end

                end % for index_measurement_sel = 1:numel( operators_born( index_object ).indices_measurement_sel )

                % create array of signal matrices
                if all( cellfun( @( x ) strcmp( class( x ), 'processing.signal_matrix' ), u_M{ index_object } ) )
                    u_M{ index_object } = cat( 1, u_M{ index_object }{ : } );
                end

            end % for index_object = 1:numel( operators_born )

            % avoid cell array for single operators_born
            if isscalar( operators_born )
                u_M = u_M{ 1 };
            end

        end % function u_M = format_voltages( operators_born, u_M )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Hidden)

        %------------------------------------------------------------------
        % received energy (scalar; decomposition)
        %------------------------------------------------------------------
        function E_M = energy_rx_scalar( operator_born, LT, LTs_tgc_measurement )
% TODO
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.operator (scalar) for operator_born
            % calling function ensures class linear_transforms.linear_transform (scalar) for LT
            % calling function ensures class linear_transforms.concatenations.diagonal (cell) for LTs_tgc_measurement

            % extract individual dictionaries for concatenation
            if isa( LT, 'linear_transforms.concatenations.vertical' )
                LT = LT.transforms;
            end

            % ensure cell array for LT
            if ~iscell( LT )
                LT = { LT };
            end

            %--------------------------------------------------------------
            % 2.) received energy (decomposition into unique parts)
            %--------------------------------------------------------------
            % unique indices of selected sequential pulse-echo measurements
            indices_measurement_sel = operator_born.indices_measurement_sel;

            % create common format string for filename
            str_format_common = sprintf( 'data/%s/setup_%%s/E_M_settings_%%s_TGC_%%s_options_aliasing_%%s', operator_born.sequence.setup.str_name );

            E_M_unique = cell( size( LT ) );

            % iterate dictionaries
            for index_dictionary = 1:numel( LT )

                % initialize unique received energies w/ zeros
                E_M_unique{ index_dictionary } = physical_values.squarevolt( zeros( LT{ index_dictionary }.N_coefficients, numel( indices_measurement_sel ) ) );

                % check dictionary
                if isa( LT{ index_dictionary }, 'linear_transforms.identity' )

                    %----------------------------------------------
                    % ii.) canonical basis
                    %----------------------------------------------
                    % create format string for filename
                    str_format = sprintf( '%s.mat', str_format_common );

                    % iterate unique selected sequential pulse-echo measurements
                    for index_measurement_sel = 1:numel( indices_measurement_sel )

                        % index of sequential pulse-echo measurement
                        index_measurement = indices_measurement_sel( index_measurement_sel );

                        % set momentary scattering operator options
                        options_momentary_act = set_properties( operator_born.options.momentary, scattering.options.sequence_selected( index_measurement ) );
                        operator_born = set_options_momentary( operator_born, options_momentary_act );

                        % load or compute received energies (canonical basis)
                        E_M_unique{ index_dictionary }( :, index_measurement_sel ) ...
                        = auxiliary.compute_or_load_hash( str_format, @energy_rx_canonical, [ 3, 4, 2, 5 ], [ 1, 2 ], ...
                            operator_born, LTs_tgc_measurement( index_measurement_sel ), ...
                            { operator_born.sequence.setup.xdc_array.aperture, operator_born.sequence.setup.homogeneous_fluid, operator_born.sequence.setup.FOV, operator_born.sequence.setup.str_name }, ...
                            operator_born.sequence.settings( index_measurement ), ...
                            operator_born.options.momentary.anti_aliasing );

                    end % for index_measurement_sel = 1:numel( indices_measurement_sel )

                else

                    %------------------------------------------------------
                    % ii.) arbitrary linear transform
                    %------------------------------------------------------
                    % create format string for filename
                    str_format = sprintf( '%s_transform_%%s.mat', str_format_common );

                    % iterate unique selected sequential pulse-echo measurements
                    for index_measurement_sel = 1:numel( indices_measurement_sel )

                        % index of sequential pulse-echo measurement
                        index_measurement = indices_measurement_sel( index_measurement_sel );

                        % set momentary scattering operator options
                        options_momentary_act = set_properties( operator_born.options.momentary, scattering.options.sequence_selected( index_measurement ) );
                        operator_born = set_options_momentary( operator_born, options_momentary_act );

                        % load or compute received energies (arbitrary linear transform)
                        E_M_unique{ index_dictionary }( :, index_measurement_sel ) ...
                        = auxiliary.compute_or_load_hash( str_format, @energy_rx_arbitrary, [ 4, 5, 3, 6, 2 ], [ 1, 2, 3 ], ...
                                operator_born, LT{ index_dictionary }, LTs_tgc_measurement( index_measurement_sel ), ...
                                { operator_born.sequence.setup.xdc_array.aperture, operator_born.sequence.setup.homogeneous_fluid, operator_born.sequence.setup.FOV, operator_born.sequence.setup.str_name }, ...
                                operator_born.sequence.settings( index_measurement ), ...
                                operator_born.options.momentary.anti_aliasing );

                    end % for index_measurement_sel = 1:numel( indices_measurement_sel )

                end % if isa( LT{ index_dictionary }, 'linear_transforms.identity' )

                %----------------------------------------------------------
                % iii.) sum unique received energies
                %----------------------------------------------------------
                E_M_unique{ index_dictionary } = sum( E_M_unique{ index_dictionary }, 2 );

            end % for index_dictionary = 1:numel( LT )

            % concatenate vertically
            E_M = cat( 1, E_M_unique{ : } );

        end % function E_M = energy_rx_scalar( operator_born, LT, LTs_tgc_measurement )

	end % methods (Hidden)

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
                indices_f_measurement_to_global = operator_born.sequence.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                indices_f_mix_to_measurement = operator_born.sequence.settings( index_measurement ).indices_f_to_unique;

                % extract occupied grid points from incident pressure
                p_incident_occupied = double( operator_born.incident_waves( index_measurement ).p_incident.samples( :, indices_occupied ) );

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.sequence.prefactors{ index_measurement };

                % numbers of frequencies in mixed voltage signals
                axes_f = reshape( [ prefactors.axis ], size( prefactors ) );
                N_samples_f = abs( axes_f );

                %----------------------------------------------------------
                % ii.) compute mixed voltage signals for the active array elements
                %----------------------------------------------------------
                % specify cell arrays for u_M{ index_measurement_sel }
                u_M{ index_measurement_sel } = cell( size( operator_born.sequence.settings( index_measurement ).rx ) );

                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.sequence.settings( index_measurement ).rx )

                    %------------------------------------------------------
                    % a) active array elements and pressure field (current frequencies)
                    %------------------------------------------------------
                    % number of active array elements
                    N_elements_active = numel( operator_born.sequence.settings( index_measurement ).rx( index_mix ).indices_active );

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
                        index_element = operator_born.sequence.settings( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operator_born.sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.sequence.setup.indices_grid_FOV_shift( indices_occupied, index_element );

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
                            h_rx = transfer_function( operator_born.sequence.setup, axes_f( index_mix ), index_element );

                            % apply spatial anti-aliasing filter
                            h_rx = anti_aliasing_filter( operator_born.sequence.setup, h_rx, operator_born.options.momentary.anti_aliasing );
                            h_rx = double( h_rx.samples );

                        end % if isa( operator_born.sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                        %--------------------------------------------------
                        % compute matrix-vector product and mix voltage signals
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_occupied_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        u_M{ index_measurement_sel }{ index_mix } = u_M{ index_measurement_sel }{ index_mix } + Phi_act * fluctuations( indices_occupied, : );

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.sequence.settings( index_measurement ).rx )

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
            gamma_hat = zeros( operator_born.sequence.setup.FOV.shape.grid.N_points, size( u_M, 2 ) );

            % partition matrix into cell arrays
            N_obs = { operator_born.sequence.settings.N_observations };
            u_M = mat2cell( u_M, cellfun( @( x ) sum( x( : ) ), N_obs( operator_born.indices_measurement_sel ) ), size( u_M, 2 ) );

            % iterate selected sequential pulse-echo measurements
            for index_measurement_sel = 1:numel( operator_born.indices_measurement_sel )

                % index of sequential pulse-echo measurement
                index_measurement = operator_born.indices_measurement_sel( index_measurement_sel );

                %----------------------------------------------------------
                % a) frequency maps and prefactors
                %----------------------------------------------------------
                % map unique frequencies of pulse-echo measurement to global unique frequencies
                indices_f_measurement_to_global = operator_born.sequence.indices_f_to_unique{ index_measurement };

                % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                indices_f_mix_to_measurement = operator_born.sequence.settings( index_measurement ).indices_f_to_unique;

                % extract prefactors for all mixes (current frequencies)
                prefactors = operator_born.sequence.prefactors{ index_measurement };

                % partition matrix into cell arrays
                u_M{ index_measurement_sel } = mat2cell( u_M{ index_measurement_sel }, operator_born.sequence.settings( index_measurement ).N_observations, size( u_M{ index_measurement_sel }, 2 ) );

                % iterate mixed voltage signals
                for index_mix = 1:numel( operator_born.sequence.settings( index_measurement ).rx )

                    % number of active array elements
                    N_elements_active = numel( operator_born.sequence.settings( index_measurement ).rx( index_mix ).indices_active );

                    % extract incident acoustic pressure field for current frequencies
                    p_incident_act = double( operator_born.incident_waves( index_measurement ).p_incident.samples );
                    if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.incident_waves( index_measurement ).p_incident.axis )
                        p_incident_act = p_incident_act( indices_f_mix_to_measurement{ index_mix }, : );
                    end

                    % iterate active array elements
                    for index_active = 1:N_elements_active

                        % index of active array element
                        index_element = operator_born.sequence.settings( index_measurement ).rx( index_mix ).indices_active( index_active );

                        % spatial transfer function of the active array element
                        if isa( operator_born.sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                            %----------------------------------------------
                            % a) symmetric spatial discretization based on orthogonal regular grids
                            %----------------------------------------------
                            % shift reference spatial transfer function to infer that of the active array element
                            indices_occupied_act = operator_born.sequence.setup.indices_grid_FOV_shift( :, index_element );

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
                            h_rx = transfer_function( operator_born.sequence.setup, axes_f( index_mix ), index_element );

                            % apply spatial anti-aliasing filter
                            h_rx = anti_aliasing_filter( operator_born.sequence.setup, h_rx, operator_born.options.momentary.anti_aliasing );
                            h_rx = double( h_rx.samples );

                        end % if isa( operator_born.sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                        %--------------------------------------------------
                        % compute matrix-vector product
                        %--------------------------------------------------
                        Phi_act = h_rx .* p_incident_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                        gamma_hat = gamma_hat + Phi_act' * u_M{ index_measurement_sel }{ index_mix };

                    end % for index_active = 1:N_elements_active

                end % for index_mix = 1:numel( operator_born.sequence.settings( index_measurement ).rx )

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
        % single received energy (arbitrary linear transform)
        %------------------------------------------------------------------
        function E_M = energy_rx_arbitrary( operator_born, LT, varargin )

            % internal constant (adjust to capabilities of GPU)
            N_objects = 1024;

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing received energies (Born approximation, kappa, arbitrary linear transform)... ', str_date_time );

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
            if ~( isa( LT, 'linear_transforms.linear_transform' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be a single linear_transforms.linear_transform!';
                errorStruct.identifier = 'energy_rx:NoSingleLinearTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute received energies (arbitrary linear transform)
            %--------------------------------------------------------------
            % compute number of batches and objects in last batch
            N_batches = ceil( LT.N_coefficients / N_objects );
            N_objects_last = LT.N_coefficients - ( N_batches - 1 ) * N_objects;

            % partition indices of transform coefficients into N_batches batches
            indices_coeff = mat2cell( ( 1:LT.N_coefficients ), 1, [ N_objects * ones( 1, N_batches - 1 ), N_objects_last ] );

            % initialize received energies with zeros
            E_M = zeros( LT.N_coefficients, 1 );

            % name for temporary file
            str_filename = sprintf( 'data/%s/energy_rx_temp.mat', operator_born.sequence.setup.str_name );

            % get name of directory
            [ str_name_dir, ~, ~ ] = fileparts( str_filename );

            % ensure existence of folder str_name_dir
            [ success, errorStruct.message, errorStruct.identifier ] = mkdir( str_name_dir );
            if ~success
                error( errorStruct );
            end

            % initialize elapsed times with zero
            seconds_per_batch = zeros( 1, N_batches );

            % iterate batches of transform coefficients
            for index_batch = 1:N_batches

                % print progress in percent
                if index_batch > 1
                    N_bytes = fprintf( '%5.1f %% (elapsed: %d min. | remaining: %d min. | mean: %.2f s | last: %.2f s)', ( index_batch - 1 ) / N_batches * 1e2, round( toc( time_start ) / 60 ), round( ( N_batches - index_batch + 1 ) * mean( seconds_per_batch( 1:(index_batch - 1) ) ) / 60 ), mean( seconds_per_batch( 1:(index_batch - 1) ) ), seconds_per_batch( index_batch - 1 ) );
                else
                    N_bytes = fprintf( '%5.1f %% (elapsed: %d min.)', 0, round( toc( time_start ) / 60 ) );
                end

                %----------------------------------------------------------
                % a) create batch of coefficient vectors
                %----------------------------------------------------------
                % indices of transform coefficients
                indices_theta = ( 0:( numel( indices_coeff{ index_batch } ) - 1 ) ) * LT.N_coefficients + indices_coeff{ index_batch };

                % initialize transform coefficients
                theta_kappa = zeros( LT.N_coefficients, numel( indices_coeff{ index_batch } ) );
                theta_kappa( indices_theta ) = 1;

                %----------------------------------------------------------
                % b) quick forward scattering and received energies
                %----------------------------------------------------------
                % quick forward scattering
%                 profile on
                time_batch_start = tic;
                u_M = forward_quick( operator_born, theta_kappa, LT, varargin{ : } );
                seconds_per_batch( index_batch ) = toc( time_batch_start );
%                 profile viewer

                % compute received energy
                E_M( indices_coeff{ index_batch } ) = vecnorm( u_M, 2, 1 ).^2;

                %----------------------------------------------------------
                % c) save and display intermediate results
                %----------------------------------------------------------
                % save intermediate results
                save( str_filename, 'E_M' );

                % display intermediate results
                figure( 999 );
                imagesc( illustration.dB( squeeze( reshape( E_M, operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) )', 10 ), [ -60, 0 ] );

                % erase progress in percent
                fprintf( repmat( '\b', [ 1, N_bytes ] ) );

            end % for index_batch = 1:N_batches

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_M = energy_rx_arbitrary( operator_born, LT, varargin )

        %------------------------------------------------------------------
        % single received energy (canonical basis)
        %------------------------------------------------------------------
        function E_M = energy_rx_canonical( operator_born, varargin )

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
                errorStruct.identifier = 'energy_rx_canonical:NoSingleOperatorBorn';
                error( errorStruct );
            end

            % ensure scalar pulse-echo measurement index
            if ~isscalar( operator_born.indices_measurement_sel )
                errorStruct.message = 'operator_born.indices_measurement_sel must be a single index!';
                errorStruct.identifier = 'energy_rx_canonical:NoSingleMeasurementIndex';
                error( errorStruct );
            end

            % extract index of sequential pulse-echo measurement
            index_measurement = operator_born.indices_measurement_sel;

            % ensure nonempty LT_tgc
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                LT_tgc = varargin{ 1 };
            else
                LT_tgc = linear_transforms.identity( operator_born.sequence.setup.FOV.shape.grid.N_points );
            end % if nargin >= 2 && ~isempty( varargin{ 1 } )

            % ensure class linear_transforms.concatenations.diagonal (scalar)
            if ~( ( isa( LT_tgc, 'linear_transforms.identity' ) || isa( LT_tgc, 'linear_transforms.concatenations.diagonal' ) ) && isscalar( LT_tgc ) )
                errorStruct.message = 'Nonempty varargin{ 1 } must be a single linear_transforms.identity or linear_transforms.concatenations.diagonal!';
                errorStruct.identifier = 'energy_rx_canonical:NoSingleDiagonalConcatenation';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute energies of mixed voltage signals
            %--------------------------------------------------------------
            % map unique frequencies of pulse-echo measurement to global unique frequencies
            indices_f_measurement_to_global = operator_born.sequence.indices_f_to_unique{ index_measurement };

            % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
            indices_f_mix_to_measurement = operator_born.sequence.settings( index_measurement ).indices_f_to_unique;

            % extract prefactors for all mixes (current frequencies)
            prefactors = operator_born.sequence.prefactors{ index_measurement };

            % numbers of frequencies in mixed voltage signals
            axes_f = reshape( [ prefactors.axis ], size( prefactors ) );
            N_samples_f = abs( axes_f );

            % initialize received energies with zeros
            E_M = zeros( 1, operator_born.sequence.setup.FOV.shape.grid.N_points );

            % iterate mixed voltage signals
            for index_mix = 1:numel( operator_born.sequence.settings( index_measurement ).rx )

                %----------------------------------------------------------
                % a) active array elements and pressure field (current frequencies)
                %----------------------------------------------------------
                % number of active array elements
                N_elements_active = numel( operator_born.sequence.settings( index_measurement ).rx( index_mix ).indices_active );

                % extract incident acoustic pressure field for current frequencies
                p_incident_act = double( operator_born.incident_waves( index_measurement ).p_incident.samples( indices_f_mix_to_measurement{ index_mix }, : ) );

                %----------------------------------------------------------
                % b) compute mixed voltage signals for the active array elements
                %----------------------------------------------------------
                % initialize mixed voltage signals with zeros
                Phi_M = zeros( N_samples_f( index_mix ), operator_born.sequence.setup.FOV.shape.grid.N_points );

                % iterate active array elements
                for index_active = 1:N_elements_active

                    % index of active array element
                    index_element = operator_born.sequence.settings( index_measurement ).rx( index_mix ).indices_active( index_active );

                    % spatial transfer function of the active array element
                    if isa( operator_born.sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                        %--------------------------------------------------
                        % i.) symmetric spatial discretization based on orthogonal regular grids
                        %--------------------------------------------------
                        % shift reference spatial transfer function to infer that of the active array element
                        indices_occupied_act = operator_born.sequence.setup.indices_grid_FOV_shift( :, index_element );

                        % extract current frequencies from unique frequencies
                        if numel( indices_f_mix_to_measurement{ index_mix } ) < abs( operator_born.h_ref_aa.axis )
                            h_rx = double( operator_born.h_ref_aa.samples( indices_f_measurement_to_global( indices_f_mix_to_measurement{ index_mix } ), indices_occupied_act ) );
                        else
                            h_rx = double( operator_born.h_ref_aa.samples( :, indices_occupied_act ) );
                        end

                    else

                        %--------------------------------------------------
                        % ii.) arbitrary grid
                        %--------------------------------------------------
                        % compute spatial transfer function of the active array element
                        h_rx = transfer_function( operator_born.sequence.setup, axes_f( index_mix ), index_element );

                        % apply spatial anti-aliasing filter
                        h_rx = anti_aliasing_filter( operator_born.sequence.setup, h_rx, operator_born.options.momentary.anti_aliasing );
                        h_rx = double( h_rx.samples );

                    end % if isa( operator_born.sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                    %------------------------------------------------------
                    % iii.) compute mixed voltage signals
                    %------------------------------------------------------
                    Phi_act = h_rx .* p_incident_act .* double( prefactors( index_mix ).samples( :, index_active ) );
                    Phi_M = Phi_M + Phi_act;

                end % for index_active = 1:N_elements_active

                %----------------------------------------------------------
                % c) time gain compensation (TGC)
                %----------------------------------------------------------
                if ~isa( LT_tgc, 'linear_transforms.identity' )

                    % apply time gain compensation
                    Phi_M = forward_transform( LT_tgc.transforms{ index_mix }, Phi_M );

                end % if ~isa( LT_tgc, 'linear_transforms.identity' )

                %----------------------------------------------------------
                % d) sum energies of mixed voltage signals
                %----------------------------------------------------------
                E_M = E_M + vecnorm( Phi_M, 2, 1 ).^2;

            end % for index_mix = 1:numel( operator_born.sequence.settings( index_measurement ).rx )

            % transpose result
            E_M = E_M.';

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_M = energy_rx_canonical( operator_born, varargin )

    end % methods (Access = private, Hidden)

end % classdef operator_born < scattering.operator
