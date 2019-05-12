%
% superclass for all transducer control settings in synthesis mode
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-05-11
%
classdef setting_tx < controls.setting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        excitation_voltages ( 1, : ) discretizations.signal_matrix	% voltages exciting the active channels

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_tx( indices_active, impulse_responses, excitation_voltages )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if nargin == 0
                indices_active = 1;
                impulse_responses = discretizations.signal_matrix( math.sequence_increasing_regular( 0, 0, physical_values.second ), 1 );
                excitation_voltages = discretizations.signal_matrix( math.sequence_increasing_regular( 0, 0, physical_values.second ), physical_values.volt );
            end

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % ensure cell array for impulse_responses
            if ~iscell( impulse_responses )
                impulse_responses = { impulse_responses };
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
            objects@controls.setting( indices_active, impulse_responses );

            %--------------------------------------------------------------
            % 3.) check and set independent properties
            %--------------------------------------------------------------
            % iterate transducer control settings in synthesis mode
            for index_object = 1:numel( objects )

                switch class( excitation_voltages{ index_object } )

                    case 'discretizations.signal'

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

                    case 'discretizations.signal_matrix'

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

        end % function objects = setting_tx( indices_active, impulse_responses, excitation_voltages )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function settings_tx = discretize( settings_tx, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of arguments
            if nargin ~= 3
                errorStruct.message     = 'Three arguments are required!';
                errorStruct.identifier	= 'discretize:NumberArguments';
                error( errorStruct );
            end

            % multiple settings_tx / single intervals_t
            if ~isscalar( settings_tx ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( settings_tx ) );
            end

            % multiple settings_tx / single intervals_f
            if ~isscalar( settings_tx ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( settings_tx ) );
            end

            % single settings_tx / multiple intervals_t
            if isscalar( settings_tx ) && ~isscalar( intervals_t )
                settings_tx = repmat( settings_tx, size( intervals_t ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings_tx, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples and coefficients
            %--------------------------------------------------------------
            % transfer behavior via superclass
            settings_tx = discretize@controls.setting( settings_tx, intervals_t, intervals_f );

            % iterate transducer control settings
            for index_object = 1:numel( settings_tx )

                % compute Fourier coefficients
                settings_tx( index_object ).excitation_voltages = fourier_coefficients( settings_tx( index_object ).excitation_voltages, intervals_t( index_object ), intervals_f( index_object ) );

                % merge transforms to ensure class signal_matrix
                settings_tx( index_object ).excitation_voltages = merge( settings_tx( index_object ).excitation_voltages );

% TODO: ensure that excitation_voltages and impulse_responses have identical axes?

            end % for index_object = 1:numel( settings_tx )

        end % function settings_tx = discretize( settings_tx, intervals_t, intervals_f )

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
                indices_unique_to_f = [];
                indices_f_to_unique = [];
                return;
            end

% TODO: check matching dimensions
% TODO: impulse_responses and excitation_voltages must be single signal matrices (ensured by Fourier transform)
% TODO: move part of functionality to superclass controls.setting

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
            samples_tf = repmat( settings_tx_in( 1 ).impulse_responses.samples( 1 ), [ numel( settings_tx_in( 1 ).indices_active ), N_samples_f_unique ] );
            samples_u_tx = repmat( settings_tx_in( 1 ).excitation_voltages.samples( 1 ), [ numel( settings_tx_in( 1 ).indices_active ), N_samples_f_unique ] );

            % iterate unique frequencies
            for index_f_unique = 1:N_samples_f_unique

                % map unique frequencies to object and frequency index
                index_object = indices_unique_to_f( index_f_unique ).index_object;
                index_f = indices_unique_to_f( index_f_unique ).index_f;

                % extract samples
                samples_tf( :, index_f_unique ) = settings_tx_in( index_object ).impulse_responses.samples( :, index_f );
                samples_u_tx( :, index_f_unique ) = settings_tx_in( index_object ).excitation_voltages.samples( :, index_f );

            end % for index_f_unique = 1:N_samples_f_unique

            % create transfer functions and excitation voltages
            indices_active_unique = settings_tx_in( 1 ).indices_active;
            impulse_responses_unique = discretizations.signal_matrix( axis_f_unique, samples_tf );
            excitation_voltages_unique = discretizations.signal_matrix( axis_f_unique, samples_u_tx );

            %--------------------------------------------------------------
            % 3.) create objects
            %--------------------------------------------------------------
            setting_tx_out = controls.setting_tx( indices_active_unique, impulse_responses_unique, excitation_voltages_unique );

        end % function [ setting_tx_out, indices_unique_to_f, indices_f_to_unique ] = unique( settings_tx_in )

	end % methods

end % classdef setting_tx < controls.setting
