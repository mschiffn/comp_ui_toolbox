%
% superclass for all calibration options
%
% author: Martin F. Schiffner
% date: 2019-06-15
% modified: 2019-11-25
%
classdef options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        duration_window_t ( 1, 1 ) physical_values.time = physical_values.second( 2.5e-6 )	% time duration of window
        time_shift_ctr ( 1, 1 ) physical_values.time = physical_values.second( 16 / 40e6 )	% time shift to waveform center
        setting_window ( 1, 1 ) auxiliary.setting_window = auxiliary.setting_window         % window settings
        index_element_lb ( 1, 1 ) { mustBeInteger, mustBePositive, mustBeNonempty } = 1     % lower bound on the element index
        index_element_ub ( 1, 1 ) { mustBeInteger, mustBePositive, mustBeNonempty } = 128   % upper bound on the element index
        factor_interp ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 30    % interpolation factor
        interval_f ( 1, 1 ) math.interval = math.interval( physical_values.hertz( 2e6 ), physical_values.hertz( 6e6 ) )	% frequency interval
        handle_absorption_model = @( x ) absorption_models.time_causal( 0, 0.5, 1, x, physical_values.hertz( 4e6 ) )	% absorption model for the lossy homogeneous fluid
        method_faces ( 1, 1 ) scattering.sequences.setups.discretizations.methods.method = scattering.sequences.setups.discretizations.methods.grid_numbers( [ 4; 53 ] ); % discretization parameters for the transducer array
        lens_thickness = physical_values.meter( 0e-3 );
        c_lens = physical_values.meter_per_second( 2000 );
        index_element_tx_ref = 1                                                            % tx element for reference pulse-echo response
        index_element_rx_ref = 1                                                            % rx element for reference pulse-echo response

        % dependent properties
        interval_window_t ( 1, 1 ) math.interval = math.interval( physical_values.second( 0 ), physical_values.second( 2.5e-6 ) );
        indices_elements ( 1, : ) double

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options( durations_window_t, time_shifts_ctr, indices_element_lb, indices_element_ub, intervals_f, handles_absorption_model, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % ensure class physical_values.time
            if ~( isa( durations_window_t, 'physical_values.time' ) && isa( time_shifts_ctr, 'physical_values.time' ) )
                errorStruct.message = 'durations_window_t and time_shifts_ctr must be physical_values.time!';
                errorStruct.identifier = 'options:NoTimes';
                error( errorStruct );
            end

            % ensure positive integers
            mustBeInteger( indices_element_lb );
            mustBePositive( indices_element_lb );

            % ensure positive integers
            mustBeInteger( indices_element_ub );
            mustBePositive( indices_element_ub );

            % ensure class math.interval with physical units of frequency
            if ~( isa( intervals_f, 'math.interval' ) && all( cellfun( @( x ) isa( x, 'physical_values.frequency' ), { intervals_f.lb } ) ) )
                errorStruct.message = 'intervals_f must be math.interval whose bounds are physical_values.frequency!';
                errorStruct.identifier = 'estimate_PER_point:NoFrequencyIntervals';
                error( errorStruct );
            end

            % ensure cell array for handles_absorption_model
            if ~iscell( handles_absorption_model )
                handles_absorption_model = { handles_absorption_model };
            end

            % ensure nonempty method_faces
            if nargin >= 7 && isa( varargin{ 1 }, 'scattering.sequences.setups.discretizations.methods.method' )
                method_faces = varargin{ 1 };
            else
                method_faces = repmat( scattering.sequences.setups.discretizations.methods.grid_numbers( [ 4; 53 ] ), size( durations_window_t ) );
            end

            % ensure nonempty setting_window
            if nargin >= 8 && isa( varargin{ 2 }, 'auxiliary.setting_window' )
                setting_window = varargin{ 2 };
            else
                setting_window = auxiliary.setting_window;
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( durations_window_t, time_shifts_ctr, indices_element_lb, indices_element_ub, intervals_f, handles_absorption_model, method_faces );

            %--------------------------------------------------------------
            % 2.) create calibration options
            %--------------------------------------------------------------
            % repeat default calibration options
            objects = repmat( objects, size( durations_window_t ) );

            % iterate calibration options
            for index_object = 1:numel( objects )

                % ensure class function_handle
                if ~isa( handles_absorption_model{ index_object }, 'function_handle' )
                    errorStruct.message = sprintf( 'handles_absorption_model{ %d } must be function_handle!', index_object );
                    errorStruct.identifier = 'estimate_PER_point:NoFunctionHandles';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).duration_window_t = durations_window_t( index_object );
                objects( index_object ).time_shift_ctr = time_shifts_ctr( index_object );
                objects( index_object ).index_element_lb = indices_element_lb( index_object );
                objects( index_object ).index_element_ub = indices_element_ub( index_object );
                objects( index_object ).interval_f = intervals_f( index_object );
                objects( index_object ).handle_absorption_model = handles_absorption_model{ index_object };
                objects( index_object ).method_faces = method_faces( index_object );

                % set dependent properties
                objects( index_object ).interval_window_t = math.interval( physical_values.second( 0 ), objects( index_object ).duration_window_t );
                objects( index_object ).indices_elements = ( objects( index_object ).index_element_lb:objects( index_object ).index_element_ub );

            end % for index_object = 1:numel( objects )

        end % function objects = options( durations_window_t, time_shifts_ctr, indices_element_lb, indices_element_ub, intervals_f, handles_absorption_model, varargin )

	end % methods

end % classdef patch
