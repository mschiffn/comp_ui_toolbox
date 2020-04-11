%
% superclass for all control settings in synthesis mode
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2020-04-07
%
classdef tx < scattering.sequences.settings.controls.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        excitation_voltages ( 1, : ) processing.signal_matrix	% voltages exciting the active channels

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tx( indices_active, impulse_responses, excitation_voltages )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if nargin == 0
                indices_active = 1;
                impulse_responses = processing.signal_matrix( math.sequence_increasing_regular_quantized( 0, 0, physical_values.second ), 1 );
                excitation_voltages = processing.signal_matrix( math.sequence_increasing_regular_quantized( 0, 0, physical_values.second ), physical_values.volt );
            end

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % ensure cell array for excitation_voltages
            if ~iscell( excitation_voltages )
                excitation_voltages = { excitation_voltages };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( indices_active, excitation_voltages );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@scattering.sequences.settings.controls.common( indices_active, impulse_responses );

            %--------------------------------------------------------------
            % 3.) check and set independent properties
            %--------------------------------------------------------------
            % iterate transducer control settings in synthesis mode
            for index_object = 1:numel( objects )

                switch class( excitation_voltages{ index_object } )

                    case 'processing.signal'

                        % multiple indices_active{ index_object } / single excitation_voltages{ index_object }
                        if ~isscalar( indices_active{ index_object } ) && isscalar( excitation_voltages{ index_object } )
                            excitation_voltages{ index_object } = repmat( excitation_voltages{ index_object }, size( indices_active{ index_object } ) );
                        end

                        % ensure equal number of dimensions and sizes of cell array contents
                        auxiliary.mustBeEqualSize( indices_active{ index_object }, excitation_voltages{ index_object } );

                        % try to merge compatible signals into a single signal matrix
                        try
                            excitation_voltages{ index_object } = merge( excitation_voltages{ index_object } );
                        catch
                        end

                    case 'processing.signal_matrix'

                        % ensure single signal matrix of correct size
                        if ~isscalar( excitation_voltages{ index_object } ) || ( numel( indices_active{ index_object } ) ~= excitation_voltages{ index_object }.N_signals )
                            errorStruct.message = sprintf( 'excitation_voltages{ %d } must be a scalar and contain %d signals!', index_object, numel( indices_active{ index_object } ) );
                            errorStruct.identifier = 'setting:SizeMismatch';
                            error( errorStruct );
                        end

                end % switch class( excitation_voltages{ index_object } )

                % ensure class physical_values.volt
%                 if ~isa( [ excitation_voltages{ index_object }.samples ], 'physical_values.volt' )
%                     errorStruct.message = sprintf( 'excitation_voltages{ %d }.samples has to be physical_values.volt!', index_object );
%                     errorStruct.identifier = 'setting:NoVoltages';
%                     error( errorStruct );
%                 end

                % set independent properties
                objects( index_object ).excitation_voltages = excitation_voltages{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = tx( indices_active, impulse_responses, excitation_voltages )

        %------------------------------------------------------------------
        % spectral discretization (overload discretize method)
        %------------------------------------------------------------------
        function settings_tx = discretize( settings_tx, Ts_ref, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of arguments
            narginchk( 3, 3 );

            % multiple settings_tx / single Ts_ref
            if ~isscalar( settings_tx ) && isscalar( Ts_ref )
                Ts_ref = repmat( Ts_ref, size( settings_tx ) );
            end

            % multiple settings_tx / single intervals_f
            if ~isscalar( settings_tx ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( settings_tx ) );
            end

            % single settings_tx / multiple Ts_ref
            if isscalar( settings_tx ) && ~isscalar( Ts_ref )
                settings_tx = repmat( settings_tx, size( Ts_ref ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings_tx, Ts_ref, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples and coefficients
            %--------------------------------------------------------------
            % transfer behavior via superclass
            settings_tx = discretize@scattering.sequences.settings.controls.common( settings_tx, Ts_ref, intervals_f );

            % iterate transducer control settings
            for index_object = 1:numel( settings_tx )

                % compute Fourier coefficients
                settings_tx( index_object ).excitation_voltages = fourier_coefficients( settings_tx( index_object ).excitation_voltages, Ts_ref( index_object ), intervals_f( index_object ) );

                % merge transforms to ensure class signal_matrix
                settings_tx( index_object ).excitation_voltages = merge( settings_tx( index_object ).excitation_voltages );

                % ensure that settings_tx( index_object ).excitation_voltages and settings_tx( index_object ).impulse_responses have identical frequency axes
                if ~isequal( settings_tx( index_object ).excitation_voltages.axis, settings_tx( index_object ).impulse_responses.axis )
                    errorStruct.message = sprintf( 'Excitation voltages and impulse responses in settings_tx( %d ) must have identical frequency axes!', index_object );
                    errorStruct.identifier = 'discretize:MismatchFrequencyAxis';
                    error( errorStruct );
                end

            end % for index_object = 1:numel( settings_tx )

        end % function settings_tx = discretize( settings_tx, Ts_ref, intervals_f )

        %------------------------------------------------------------------
        % unique values in array (overload unique function)
        %------------------------------------------------------------------
        function [ setting_tx_out, indices_unique_to_f, indices_f_to_unique ] = unique( settings_tx_in )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return settings_tx_in if only one argument
            if isscalar( settings_tx_in )
                setting_tx_out = settings_tx_in;
                [ ~, indices_unique_to_f, indices_f_to_unique ] = unique( settings_tx_in.excitation_voltages.axis );
                return;
            end

% TODO: check matching dimensions
% TODO: impulse_responses and excitation_voltages must be single signal matrices (ensured by Fourier transform)
% TODO: move part of functionality to superclass scattering.sequences.settings.controls.common

            %--------------------------------------------------------------
            % 2.) extract transfer functions and excitation voltages for unique frequencies
            %--------------------------------------------------------------
            % extract frequency axes
            axes_f = repmat( settings_tx_in( 1 ).excitation_voltages.axis, size( settings_tx_in ) );
            for index_object = 2:numel( settings_tx_in )
                axes_f( index_object ) = settings_tx_in( index_object ).excitation_voltages.axis;
            end

            % extract unique frequency axis
            [ axis_f_unique, indices_unique_to_f, indices_f_to_unique ] = unique( axes_f );
            N_samples_f_unique = numel( indices_unique_to_f );

            % initialize unique samples
            samples_tf = repmat( settings_tx_in( 1 ).impulse_responses.samples( 1 ), [ N_samples_f_unique, numel( settings_tx_in( 1 ).indices_active ) ] );
            samples_u_tx = repmat( settings_tx_in( 1 ).excitation_voltages.samples( 1 ), [ N_samples_f_unique, numel( settings_tx_in( 1 ).indices_active ) ] );

            % iterate unique frequencies
            for index_f_unique = 1:N_samples_f_unique

                % map unique frequencies to object and frequency index
                index_object = indices_unique_to_f( index_f_unique ).index_object;
                index_f = indices_unique_to_f( index_f_unique ).index_f;

                % extract samples
                samples_tf( index_f_unique, : ) = settings_tx_in( index_object ).impulse_responses.samples( index_f, : );
                samples_u_tx( index_f_unique, : ) = settings_tx_in( index_object ).excitation_voltages.samples( index_f, : );

            end % for index_f_unique = 1:N_samples_f_unique

            % create transfer functions and excitation voltages
            indices_active_unique = settings_tx_in( 1 ).indices_active;
            impulse_responses_unique = processing.signal_matrix( axis_f_unique, samples_tf );
            excitation_voltages_unique = processing.signal_matrix( axis_f_unique, samples_u_tx );

            %--------------------------------------------------------------
            % 3.) create objects
            %--------------------------------------------------------------
            setting_tx_out = scattering.sequences.settings.controls.tx( indices_active_unique, impulse_responses_unique, excitation_voltages_unique );

        end % function [ setting_tx_out, indices_unique_to_f, indices_f_to_unique ] = unique( settings_tx_in )

        %------------------------------------------------------------------
        % unique deltas
        %------------------------------------------------------------------
        function deltas = unique_deltas( settings_tx )

            % extract unique deltas from impulse_responses via superclass
            deltas_impulse = unique_deltas@scattering.sequences.settings.controls.common( settings_tx );

            % extract excitation_voltages
            excitation_voltages = reshape( { settings_tx.excitation_voltages }, size( settings_tx ) );

            % specify cell array for deltas
            deltas = cell( size( settings_tx ) );

            % iterate transducer control settings
            for index_setting = 1:numel( settings_tx )
% TODO: math.sequence_increasing_regular sufficient?
                % ensure equal subclasses of math.sequence_increasing_regular_quantized
                auxiliary.mustBeEqualSubclasses( 'math.sequence_increasing_regular_quantized', excitation_voltages{ index_setting }.axis );

                % extract regular axes
                axes = reshape( [ excitation_voltages{ index_setting }.axis ], size( excitation_voltages{ index_setting } ) );

                % ensure equal subclasses of physical_values.physical_quantity
                auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', axes.delta );

                % extract deltas as row vector
                deltas{ index_setting } = reshape( [ axes.delta ], size( excitation_voltages{ index_setting } ) );

            end % for index_setting = 1:numel( settings_tx )

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', deltas_impulse, deltas{ : } );

            % extract unique deltas
            deltas = unique( [ deltas_impulse, cat( 2, deltas{ : } ) ] );

        end % function deltas = unique_deltas( settings_tx )

        %------------------------------------------------------------------
        % compute normal velocities
        %------------------------------------------------------------------
        function v_d = compute_normal_velocities( settings_tx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal_matrix
            if ~( isa( [ settings_tx.excitation_voltages ], 'processing.signal_matrix' ) && isa( [ settings_tx.impulse_responses ], 'processing.signal_matrix' ) )
                errorStruct.message = 'excitation_voltages and impulse_responses must be processing.signal_matrix!';
                errorStruct.identifier = 'compute_normal_velocities:NoSignalMatrices';
                error( errorStruct );
            end
% TODO: check for frequency domain!
            %--------------------------------------------------------------
            % 2.) compute normal velocities
            %--------------------------------------------------------------
            % extract excitation voltages and transfer functions for each transducer control settings in synthesis mode
            excitation_voltages = reshape( [ settings_tx.excitation_voltages ], size( settings_tx ) );
            impulse_responses = reshape( [ settings_tx.impulse_responses ], size( settings_tx ) );

            % compute velocities for each transducer control settings in synthesis mode
            v_d = excitation_voltages .* impulse_responses;

        end % function v_d = compute_normal_velocities( settings_tx )

	end % methods

end % classdef tx < scattering.sequences.settings.controls.common
