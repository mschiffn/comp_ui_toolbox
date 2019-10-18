%
% superclass for all scattering operators based on the Born approximation
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-08-28
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
            if nargin >= 3 && ~isempty( varargin{ 1 } )

                % ensure class linear_transforms.linear_transform (scalar)
                if ~( isa( varargin{ 1 }, 'linear_transforms.linear_transform' ) && isscalar( varargin{ 1 } ) )
                    errorStruct.message = 'Nonempty varargin{ 1 } must be a single linear_transforms.linear_transform!';
                    errorStruct.identifier = 'forward_quick:NoSingleLinearTransform';
                    error( errorStruct );
                end

% TODO: check compatibility  && isequal( operator_born.discretization.spatial.FOV.shape.grid.N_points_axis, varargin{ 1 }.N_lattice )

                % apply adjoint linear transform
                fluctuations = operator_transform( varargin{ 1 }, fluctuations, 2 );

            end % if nargin >= 3 && ~isempty( varargin{ 1 } )

            %--------------------------------------------------------------
            % 3.) compute mixed voltage signals
            %--------------------------------------------------------------
            if isa( operator_born.options.momentary.gpu, 'scattering.options.gpu_off' )
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
            if isa( operator_born.options.momentary.gpu, 'scattering.options.gpu_off' )
                gamma_hat = adjoint_quick_cpu( operator_born, u_M );
            else
                gamma_hat = scattering.combined_quick_gpu( operator_born, 2, u_M );
%                 clear mex;
            end

            %--------------------------------------------------------------
            % 3.) forward linear transform
            %--------------------------------------------------------------
            if nargin >= 3 && ~isempty( varargin{ 1 } )

                % ensure class linear_transforms.linear_transform (scalar)
                if ~( isa( varargin{ 1 }, 'linear_transforms.linear_transform' ) && isscalar( varargin{ 1 } ) )
                    errorStruct.message = 'Nonempty varargin{ 1 } must be a single linear_transforms.linear_transform!';
                    errorStruct.identifier = 'adjoint_quick:NoSingleLinearTransform';
                    error( errorStruct );
                end

                % apply forward linear transform
                theta_hat = operator_transform( varargin{ 1 }, gamma_hat, 1 );

            else

                % use canonical basis
                theta_hat = gamma_hat;

            end % if nargin >= 3 && ~isempty( varargin{ 1 } )

            % illustrate
            temp_1 = squeeze( reshape( double( abs( gamma_hat( :, 1 ) ) ), operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) );
            temp_2 = squeeze( reshape( double( abs( theta_hat( :, 1 ) ) ), operator_born.sequence.setup.FOV.shape.grid.N_points_axis ) );
            figure(999);
            if ismatrix( temp_1 )
                subplot( 1, 2, 1 );
                imagesc( illustration.dB( temp_1, 20 )', [ -60, 0 ] );
                subplot( 1, 2, 2 );
                imagesc( illustration.dB( temp_2, 20 )', [ -60, 0 ] );
            else
                subplot( 1, 2, 1 );
                imagesc( illustration.dB( squeeze( temp_1( :, 5, : ) ), 20 )', [ -60, 0 ] );
                subplot( 1, 2, 2 );
                imagesc( illustration.dB( squeeze( temp_2( :, 5, : ) ), 20 )', [ -60, 0 ] );
            end

        end % function theta_hat = adjoint_quick( operator_born, u_M, varargin )

        %------------------------------------------------------------------
        % quick combined scattering
        %------------------------------------------------------------------
        function y = combined_quick( operator_born, x, mode, varargin )

            switch mode

                case 0
                    % return size of forward transform
                    N_observations = 0;
                    for index_setting = operator_born.indices_measurement_sel
                        N_observations = N_observations + sum( cellfun( @numel, operator_born.sequence.settings( index_setting ).indices_f_to_unique ) );
                    end % for index_setting = operator_born.indices_measurement_sel
% TODO: wrong! number of coefficients in 2nd entry
                    y = [ N_observations, operator_born.sequence.size( 2 ) ];
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
                % b) quick forward scattering
                %----------------------------------------------------------
%                 profile on
                u_M{ index_object } = forward_quick( operators_born( index_object ), fluctuations{ index_object }( : ), linear_transforms{ index_object } );
                u_M{ index_object } = physical_values.volt( u_M{ index_object } );
%                 profile viewer

                %----------------------------------------------------------
                % c) create signals or signal matrices
                %----------------------------------------------------------
                % partition matrix into cell arrays
                N_observations = { operators_born( index_object ).sequence.size( 1 ) };
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
                if all( cellfun( @( x ) strcmp( class( x ), 'discretizations.signal_matrix' ), u_M{ index_object } ) )
                    u_M{ index_object } = cat( 1, u_M{ index_object }{ : } );
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
        function [ theta_hat, rel_RMSE ] = adjoint( operators_born, u_M, varargin )

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
            if ~iscell( u_M ) || all( cellfun( @( x ) ~iscell( x ), u_M ) )
                u_M = { u_M };
            end

            % ensure nonempty linear_transforms
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                linear_transforms = varargin{ 1 };
            else
                linear_transforms = cell( size( operators_born ) );
                for index_operator = 1:numel( operators_born )
                    linear_transforms{ index_operator } = { { [] } };
                end
            end

            % ensure cell array for linear_transforms
            if ~iscell( linear_transforms ) || all( cellfun( @( x ) ~iscell( x ), linear_transforms ) )
                linear_transforms = { linear_transforms };
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
            % 2.) process scattering operators
            %--------------------------------------------------------------
            % specify cell arrays
            theta_hat = cell( size( operators_born ) );
            rel_RMSE = cell( size( operators_born ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure cell array for linear_transforms{ index_operator }
                if ~iscell( linear_transforms{ index_operator } )
                    linear_transforms{ index_operator } = linear_transforms( index_operator );
                end

                %----------------------------------------------------------
                % b) process linear transforms
                %----------------------------------------------------------
                % specify cell arrays
                theta_hat{ index_operator } = cell( size( linear_transforms{ index_operator } ) );
                rel_RMSE{ index_operator } = cell( size( linear_transforms{ index_operator } ) );

                % iterate linear transforms
                for index_transform = 1:numel( linear_transforms{ index_operator } )

                    %------------------------------------------------------
                    % i.) check arguments
                    %------------------------------------------------------
                    % set momentary scattering operator options
                    operators_born_config = set_properties_momentary( operators_born( index_operator ), varargin{ 2:end } );

                    % ensure class linear_transforms.linear_transform
                    if ~isa( linear_transforms{ index_operator }{ index_transform }, 'linear_transforms.linear_transform' )
                        errorStruct.message = sprintf( 'linear_transforms{ %d }{ %d } must be linear_transforms.linear_transform!', index_operator, index_transform );
                        errorStruct.identifier = 'adjoint:NoLinearTransforms';
                        error( errorStruct );
                    end

                    % multiple operators_born_config / single linear_transforms{ index_operator }{ index_transform }
                    if ~isscalar( operators_born_config ) && isscalar( linear_transforms{ index_operator }{ index_transform } )
                        linear_transforms{ index_operator }{ index_transform } = repmat( linear_transforms{ index_operator }{ index_transform }, size( operators_born_config ) );
                    end

                    % ensure equal number of dimensions and sizes
                    auxiliary.mustBeEqualSize( operators_born_config, linear_transforms{ index_operator }{ index_transform } );

                    %------------------------------------------------------
                    % ii.) process configurations
                    %------------------------------------------------------
                    % numbers of transform coefficients
                    N_coefficients = reshape( [ linear_transforms{ index_operator }{ index_transform }.N_coefficients ], size( linear_transforms{ index_operator }{ index_transform } ) );

                    % ensure identical numbers of transform coefficients
                    if any( N_coefficients( : ) ~= N_coefficients( 1 ) )
                        errorStruct.message = sprintf( 'linear_transforms{ %d }{ %d } must have identical numbers of transform coefficients!', index_operator, index_transform );
                        errorStruct.identifier = 'adjoint:InvalidNumbersOfCoefficients';
                        error( errorStruct );
                    end

                    % initialize adjoint coefficients w/ zeros
                    theta_hat{ index_operator }{ index_transform } = zeros( N_coefficients( 1 ), numel( operators_born_config ) );
                    rel_RMSE{ index_operator }{ index_transform } = zeros( 1, numel( operators_born_config ) );

                    % iterate configurations
                    for index_config = 1:numel( operators_born_config )

                        %--------------------------------------------------
                        % A) check mixed voltage signals
                        %--------------------------------------------------
                        % ensure class discretizations.signal_matrix
                        if ~isa( u_M{ index_operator }{ index_config }, 'discretizations.signal_matrix' )
                            errorStruct.message = sprintf( 'u_M{ %d }{ %d } must be discretizations.signal_matrix!', index_operator, index_config );
                            errorStruct.identifier = 'adjoint:NoSignalMatrices';
                            error( errorStruct );
                        end

                        %--------------------------------------------------
                        % B) normalize mixed voltage signals
                        %--------------------------------------------------
                        u_M_vect = return_vector( u_M{ index_operator }{ index_config } );
                        u_M_vect_norm = norm( u_M_vect );
                        u_M_vect_normed = u_M_vect / u_M_vect_norm;

                        %--------------------------------------------------
                        % C) quick adjoint scattering
                        %--------------------------------------------------
%                       profile on
                        theta_hat{ index_operator }{ index_transform }( :, index_config ) = adjoint_quick( operators_born_config( index_config ), u_M_vect_normed, linear_transforms{ index_operator }{ index_transform }( index_config ) );
%                       profile viewer

                        %--------------------------------------------------
                        % D) quick forward scattering and rel. RMSEs
                        %--------------------------------------------------
                        % estimate normalized mixed voltage signals
                        u_M_vect_normed_est = forward_quick( operators_born_config( index_config ), theta_hat{ index_operator }{ index_transform }( :, index_config ), linear_transforms{ index_operator }{ index_transform }( index_config ) );

                        % compute relative RMSE
                        u_M_vect_normed_res = u_M_vect_normed - u_M_vect_normed_est;
                        rel_RMSE{ index_operator }{ index_transform }( index_config ) = norm( u_M_vect_normed_res( : ), 2 );

                    end % for index_config = 1:numel( operators_born_config )

                    %------------------------------------------------------
                    % iii.) create images
                    %------------------------------------------------------
                    theta_hat{ index_operator }{ index_transform } ...
                    = discretizations.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, ...
                                             mat2cell( theta_hat{ index_operator }{ index_transform }, N_coefficients( 1 ), ones( 1, numel( operators_born_config ) ) ) );

                end % for index_transform = 1:numel( linear_transforms{ index_operator } )

                % avoid cell arrays for single linear_transforms{ index_operator }
                if isscalar( linear_transforms{ index_operator } )
                    theta_hat{ index_operator } = theta_hat{ index_operator }{ 1 };
                    rel_RMSE{ index_operator } = rel_RMSE{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators_born )

            % avoid cell arrays for single operators_born
            if isscalar( operators_born )
                theta_hat = theta_hat{ 1 };
                rel_RMSE = rel_RMSE{ 1 };
            end

        end % function [ theta_hat, rel_RMSE ] = adjoint( operators_born, u_M, varargin )

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
            if ~iscell( indices ) || all( cellfun( @( x ) ~iscell( x ), indices ) )
                indices = { indices };
            end

            % ensure nonempty linear_transforms
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                linear_transforms = varargin{ 1 };
            else
                linear_transforms = cell( size( operators_born ) );
                for index_operator = 1:numel( operators_born )
                    linear_transforms{ index_operator } = { [] };
                end
            end

            % ensure cell array for linear_transforms
            if ~iscell( linear_transforms ) || all( cellfun( @( x ) ~iscell( x ), linear_transforms ) )
                linear_transforms = { linear_transforms };
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
            % 2.) process scattering operators
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
                % ensure cell array for linear_transforms{ index_operator }
                if ~iscell( linear_transforms{ index_operator } )
                    linear_transforms{ index_operator } = { linear_transforms{ index_operator } };
                end

                % ensure cell array for indices{ index_operator }
                if ~iscell( indices{ index_operator } )
                    indices{ index_operator } = indices( index_operator );
                end

                % multiple linear_transforms{ index_operator } / single indices{ index_operator }
                if ~isscalar( linear_transforms{ index_operator } ) && isscalar( indices{ index_operator } )
                    indices{ index_operator } = repmat( indices{ index_operator }, size( linear_transforms{ index_operator } ) );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( linear_transforms{ index_operator }, indices{ index_operator } );

                %----------------------------------------------------------
                % b) process linear transforms
                %----------------------------------------------------------
                % specify cell arrays
                theta_tpsf{ index_operator } = cell( size( linear_transforms{ index_operator } ) );
                E_M{ index_operator } = cell( size( linear_transforms{ index_operator } ) );
                adjointness{ index_operator } = cell( size( linear_transforms{ index_operator } ) );

                % iterate linear transforms
                for index_transform = 1:numel( linear_transforms{ index_operator } )

                    %------------------------------------------------------
                    % i.) check arguments
                    %------------------------------------------------------
                    % set momentary scattering operator options
                    operators_born_config = set_properties_momentary( operators_born( index_operator ), varargin{ 2:end } );

                    % ensure numeric matrix
                    if ~( isnumeric( indices{ index_operator }{ index_transform } ) && ismatrix( indices{ index_operator }{ index_transform } ) )
                        errorStruct.message = sprintf( 'indices{ %d }{ %d } must be a numeric matrix!', index_operator, index_transform );
                        errorStruct.identifier = 'tpsf:NoNumericMatrix';
                        error( errorStruct );
                    end

                    % ensure valid indices{ index_operator }{ index_transform }
                    mustBeInteger( indices{ index_operator }{ index_transform } );
                    mustBePositive( indices{ index_operator }{ index_transform } );

                    % ensure class linear_transforms.linear_transform
% TODO: empty transform?
                    if ~isa( linear_transforms{ index_operator }{ index_transform }, 'linear_transforms.linear_transform' )
                        errorStruct.message = sprintf( 'linear_transforms{ %d }{ %d } must be linear_transforms.linear_transform!', index_operator, index_transform );
                        errorStruct.identifier = 'tpsf:NoLinearTransforms';
                        error( errorStruct );
                    end

                    % multiple operators_born_config / single linear_transforms{ index_operator }{ index_transform }
                    if ~isscalar( operators_born_config ) && isscalar( linear_transforms{ index_operator }{ index_transform } )
                        linear_transforms{ index_operator }{ index_transform } = repmat( linear_transforms{ index_operator }{ index_transform }, size( operators_born_config ) );
                    end

                    % ensure equal number of dimensions and sizes
                    auxiliary.mustBeEqualSize( operators_born_config, linear_transforms{ index_operator }{ index_transform } );

                    %------------------------------------------------------
                    % ii.) process configurations
                    %------------------------------------------------------
                    % number of TPSFs per configuration
                    N_tpsf = numel( indices{ index_operator }{ index_transform } );

                    % numbers of transform coefficients
                    N_coefficients = reshape( [ linear_transforms{ index_operator }{ index_transform }.N_coefficients ], size( linear_transforms{ index_operator }{ index_transform } ) );

                    % ensure identical numbers of transform coefficients
                    if any( N_coefficients( : ) ~= N_coefficients( 1 ) )
                        errorStruct.message = sprintf( 'linear_transforms{ %d }{ %d } must have identical numbers of transform coefficients!', index_operator, index_transform );
                        errorStruct.identifier = 'tpsf:InvalidNumbersOfCoefficients';
                        error( errorStruct );
                    end

                    % ensure indices{ index_operator }{ index_transform } less than or equal N_coefficients
                    if any( indices{ index_operator }{ index_transform }( : ) > N_coefficients( 1 ) )
                        errorStruct.message = sprintf( 'linear_transforms{ %d }{ %d } must be linear_transforms.linear_transform!', index_operator, index_transform );
                        errorStruct.identifier = 'tpsf:NoLinearTransforms';
                        error( errorStruct );
                    end

                    % specify cell array for theta_tpsf and initialize w/ zeros
                    theta_tpsf{ index_operator }{ index_transform } = cell( size( operators_born_config ) );
                    E_M{ index_operator }{ index_transform } = zeros( numel( operators_born_config ), N_tpsf );
                    adjointness{ index_operator }{ index_transform } = zeros( numel( operators_born_config ), N_tpsf );

                    % iterate configurations
                    for index_config = 1:numel( operators_born_config )

                        %--------------------------------------------------
                        % A) create coefficient vectors
                        %--------------------------------------------------
                        % a) indices of coefficients
                        indices_tpsf = ( 0:( N_tpsf - 1 ) ) * N_coefficients( 1 ) + indices{ index_operator }{ index_transform }( : )';

                        % b) initialize coefficient vectors
                        theta = zeros( N_coefficients( 1 ), N_tpsf );
                        theta( indices_tpsf ) = 1;

                        %--------------------------------------------------
                        % B) quick forward scattering and received energies
                        %--------------------------------------------------
                        u_M = forward_quick( operators_born_config( index_config ), theta, linear_transforms{ index_operator }{ index_transform }( index_config ) );
                        E_M{ index_operator }{ index_transform }( index_config, : ) = vecnorm( u_M, 2, 1 ).^2;

                        %--------------------------------------------------
                        % C) quick adjoint scattering and test for adjointness
                        %--------------------------------------------------
                        theta_tpsf{ index_operator }{ index_transform }{ index_config } = adjoint_quick( operators_born_config( index_config ), u_M, linear_transforms{ index_operator }{ index_transform }( index_config ) );
                        adjointness{ index_operator }{ index_transform }( index_config, : ) = E_M{ index_operator }{ index_transform }( index_config, : ) - theta_tpsf{ index_operator }{ index_transform }{ index_config }( indices_tpsf );

                    end % for index_config = 1:numel( operators_born_config )

                    %------------------------------------------------------
                    % iii.) create images
                    %------------------------------------------------------
                    theta_tpsf{ index_operator }{ index_transform } ...
                    = discretizations.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, ...
                                             theta_tpsf{ index_operator }{ index_transform } );

                end % for index_transform = 1:numel( linear_transforms{ index_operator } )

                % avoid cell array for single linear_transforms{ index_operator }
                if isscalar( linear_transforms{ index_operator } )
                    theta_tpsf{ index_operator } = theta_tpsf{ index_operator }{ 1 };
                    E_M{ index_operator } = E_M{ index_operator }{ 1 };
                    adjointness{ index_operator } = adjointness{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators_born )

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
                linear_transforms = cell( size( operators_born ) );
                for index_operator = 1:numel( operators_born )
                    linear_transforms{ index_operator } = { [] };
                end
            end

            % ensure cell array for linear_transforms
            if ~iscell( linear_transforms ) || all( cellfun( @( x ) ~iscell( x ), linear_transforms ) )
                linear_transforms = { linear_transforms };
            end

            % ensure nonempty indices_measurement_sel
            if nargin >= 3 && ~isempty( varargin{ 2 } )
                indices_measurement_sel = varargin{ 2 };
            else
                indices_measurement_sel = cell( size( operators_born ) );
                for index_operator = 1:numel( operators_born )
                    indices_measurement_sel{ index_operator } = { ( 1:numel( operators_born( index_operator ).incident_waves ) ) };
                end
            end

            % ensure cell array for indices_measurement_sel
            if ~iscell( indices_measurement_sel ) || all( cellfun( @( x ) ~iscell( x ), indices_measurement_sel ) )
                indices_measurement_sel = { indices_measurement_sel };
            end

            % multiple operators_born / single linear_transforms
            if ~isscalar( operators_born ) && isscalar( linear_transforms )
                linear_transforms = repmat( linear_transforms, size( operators_born ) );
            end

            % multiple operators_born / single indices_measurement_sel
            if ~isscalar( operators_born ) && isscalar( indices_measurement_sel )
                indices_measurement_sel = repmat( indices_measurement_sel, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born, linear_transforms, indices_measurement_sel );

            %--------------------------------------------------------------
            % 2.) compute received energies
            %--------------------------------------------------------------
            % specify cell array for E_M
            E_M = cell( size( operators_born ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators_born )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure cell array for linear_transforms{ index_operator }
                if ~iscell( linear_transforms{ index_operator } )
                    linear_transforms{ index_operator } = num2cell( linear_transforms{ index_operator } );
                end

                % ensure cell array for indices_measurement_sel{ index_operator }
                if ~iscell( indices_measurement_sel{ index_operator } ) || all( cellfun( @( x ) ~iscell( x ), indices_measurement_sel{ index_operator } ) )
                    indices_measurement_sel{ index_operator } = { indices_measurement_sel{ index_operator } };
                end

                % multiple linear_transforms{ index_operator } / single indices_measurement_sel{ index_operator }
                if ~isscalar( linear_transforms{ index_operator } ) && isscalar( indices_measurement_sel{ index_operator } )
                    indices_measurement_sel{ index_operator } = repmat( indices_measurement_sel{ index_operator }, size( linear_transforms{ index_operator } ) );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( linear_transforms{ index_operator }, indices_measurement_sel{ index_operator } );

                %----------------------------------------------------------
                % b) compute received energies
                %----------------------------------------------------------
                % specify cell array for E_M{ index_operator }
                E_M{ index_operator } = cell( size( linear_transforms{ index_operator } ) );

                % iterate linear transforms
                for index_transform = 1:numel( linear_transforms{ index_operator } )

                    %------------------------------------------------------
                    % i.) check arguments
                    %------------------------------------------------------
                    % ensure cell array for indices_measurement_sel{ index_operator }{ index_transform }
                    if ~iscell( indices_measurement_sel{ index_operator }{ index_transform } )
                        indices_measurement_sel{ index_operator }{ index_transform } = { indices_measurement_sel{ index_operator }{ index_transform } };
                    end

                    %------------------------------------------------------
                    % ii.) compute unique received energies
                    %------------------------------------------------------
                    % unique indices of selected sequential pulse-echo measurements
                    [ indices_measurement_sel_unique, ~, indices_config ] = unique( cat( 1, indices_measurement_sel{ index_operator }{ index_transform }{ : } ) );
                    indices_config = mat2cell( indices_config, cellfun( @numel, indices_measurement_sel{ index_operator }{ index_transform } ) );

% TODO: check unique transforms if they are stacked!

                    % create common format string for filename
                    str_format_common = sprintf( 'data/%s/setup_%%s/E_M_settings_%%s', operators_born( index_operator ).sequence.setup.str_name );

                    % check linear transform
                    if ~isempty( linear_transforms{ index_operator }{ index_transform } )

                        %--------------------------------------------------
                        % i.) arbitrary linear transform
                        %--------------------------------------------------
                        % initialize unique received energies w/ zeros
                        E_M_unique = physical_values.squarevolt( zeros( linear_transforms{ index_operator }{ index_transform }.N_coefficients, numel( indices_measurement_sel_unique ) ) );

                        % create format string for filename
                        str_format = sprintf( '%s_transform_%%s_options_aliasing_%%s.mat', str_format_common );

                        % iterate unique selected sequential pulse-echo measurements
                        for index_measurement_sel = 1:numel( indices_measurement_sel_unique )

                            % index of sequential pulse-echo measurement
                            index_measurement = indices_measurement_sel_unique( index_measurement_sel );

                            % set momentary scattering operator options
                            operators_born( index_operator ) = set_properties_momentary( operators_born( index_operator ), varargin{ 3:end }, scattering.options.sequence_selected( index_measurement ) );

                            % load or compute received energies (arbitrary linear transform)
                            E_M_unique( :, index_measurement_sel )...
                            = auxiliary.compute_or_load_hash( str_format, @energy_rx_arbitrary, [ 3, 4, 2, 5 ], [ 1, 2 ],...
                                operators_born( index_operator ), linear_transforms{ index_operator }{ index_transform },...
                                { operators_born( index_operator ).sequence.setup.xdc_array.aperture, operators_born( index_operator ).sequence.setup.homogeneous_fluid, operators_born( index_operator ).sequence.setup.FOV, operators_born( index_operator ).sequence.setup.str_name },...
                                operators_born( index_operator ).sequence.settings( index_measurement ),...
                                operators_born( index_operator ).options.momentary.anti_aliasing );

                        end % for index_measurement_sel = 1:numel( indices_measurement_sel_unique )

                    else

                        %--------------------------------------------------
                        % ii.) canonical basis
                        %--------------------------------------------------
                        % initialize unique received energies w/ zeros
                        E_M_unique = physical_values.squarevolt( zeros( operators_born( index_operator ).sequence.setup.FOV.shape.grid.N_points, numel( indices_measurement_sel_unique ) ) );

                        % create format string for filename
                        str_format = sprintf( '%s_options_aliasing_%%s.mat', str_format_common );

                        % iterate unique selected sequential pulse-echo measurements
                        for index_measurement_sel = 1:numel( indices_measurement_sel_unique )

                            % index of sequential pulse-echo measurement
                            index_measurement = indices_measurement_sel_unique( index_measurement_sel );

                            % set momentary scattering operator options
                            operators_born( index_operator ) = set_properties_momentary( operators_born( index_operator ), varargin{ 3:end }, scattering.options.sequence_selected( index_measurement ) );

                            % load or compute received energies (canonical basis)
                            E_M_unique( :, index_measurement_sel )...
                            = auxiliary.compute_or_load_hash( str_format, @energy_rx_canonical, [ 3, 4, 5 ], [ 1, 2 ],...
                                operators_born( index_operator ), index_measurement,...
                                { operators_born( index_operator ).sequence.setup.xdc_array.aperture, operators_born( index_operator ).sequence.setup.homogeneous_fluid, operators_born( index_operator ).sequence.setup.FOV, operators_born( index_operator ).sequence.setup.str_name },...
                                operators_born( index_operator ).sequence.settings( index_measurement ),...
                                operators_born( index_operator ).options.momentary.anti_aliasing );

                        end % for index_measurement_sel = 1:numel( indices_measurement_sel_unique )

                    end % if ~isempty( linear_transforms{ index_operator }{ index_transform } )

                    %------------------------------------------------------
                    % iii.) sum unique received energies according to config
                    %------------------------------------------------------
                    % initialize received energies w/ zeros
                    E_M{ index_operator }{ index_transform } = physical_values.squarevolt( zeros( operators_born( index_operator ).sequence.setup.FOV.shape.grid.N_points, numel( indices_config ) ) );

                    % sum received energies according to indices_config
                    for index_config = 1:numel( indices_config )

                        % sum received energies
                        E_M{ index_operator }{ index_transform }( :, index_config ) = sum( E_M_unique( :, indices_config{ index_config } ), 2 );

                    end % for index_config = 1:numel( indices_config )
                
                end % for index_transform = 1:numel( linear_transforms{ index_operator } )

                % avoid cell array for single linear_transforms{ index_operator }
                if isscalar( linear_transforms{ index_operator } )
                    E_M{ index_operator } = E_M{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators_born )

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
        function E_M = energy_rx_arbitrary( operator_born, linear_transform )

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
            if ~( isa( linear_transform, 'linear_transforms.linear_transform' ) && isscalar( linear_transform ) )
                errorStruct.message = 'linear_transform must be a single linear_transforms.linear_transform!';
                errorStruct.identifier = 'energy_rx:NoSingleLinearTransform';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute received energies (arbitrary linear transform)
            %--------------------------------------------------------------
            % compute number of batches and objects in last batch
            N_batches = ceil( linear_transform.N_coefficients / N_objects );
            N_objects_last = linear_transform.N_coefficients - ( N_batches - 1 ) * N_objects;

            % partition indices of transform coefficients into N_batches batches
            indices_coeff = mat2cell( ( 1:linear_transform.N_coefficients ), 1, [ N_objects * ones( 1, N_batches - 1 ), N_objects_last ] );

            % initialize received energies with zeros
            E_M = zeros( linear_transform.N_coefficients, 1 );

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
                indices_theta = ( 0:( numel( indices_coeff{ index_batch } ) - 1 ) ) * linear_transform.N_coefficients + indices_coeff{ index_batch };

                % initialize transform coefficients
                theta_kappa = zeros( linear_transform.N_coefficients, numel( indices_coeff{ index_batch } ) );
                theta_kappa( indices_theta ) = 1;

                %----------------------------------------------------------
                % b) quick forward scattering and received energies
                %----------------------------------------------------------
                % quick forward scattering
                time_batch_start = tic;
                u_M = forward_quick( operator_born, theta_kappa, linear_transform );
                seconds_per_batch( index_batch ) = toc( time_batch_start );

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

        end % function E_M = energy_rx_arbitrary( operator_born, linear_transform )

        %------------------------------------------------------------------
        % single received energy (canonical basis)
        %------------------------------------------------------------------
        function E_M = energy_rx_canonical( operator_born, index_measurement )

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
% TODO: extract from options
            % ensure positive integer less than or equal to the number of sequential pulse-echo measurements
            mustBePositive( index_measurement );
            mustBeInteger( index_measurement );
            mustBeLessThanOrEqual( index_measurement, numel( operator_born.incident_waves ) );

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
                % c) sum energies of mixed voltage signals
                %----------------------------------------------------------
                E_M = E_M + vecnorm( Phi_M, 2, 1 ).^2;

            end % for index_mix = 1:numel( operator_born.sequence.settings( index_measurement ).rx )

            % transpose result
            E_M = E_M.';

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_M = energy_rx_canonical( operator_born, index_measurement )

    end % methods (Access = private, Hidden)

end % classdef operator_born < scattering.operator
