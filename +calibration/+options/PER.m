%
% superclass for all pulse-echo response estimation options
%
% author: Martin F. Schiffner
% date: 2019-11-26
% modified: 2019-11-26
%
classdef PER < calibration.options.common

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        interval_f ( 1, 1 ) math.interval { mustBeNonempty } = math.interval( physical_values.hertz( 2e6 ), physical_values.hertz( 6e6 ) )	% frequency interval
        handle_absorption_model ( 1, 1 ) function_handle { mustBeNonempty } = @( x ) absorption_models.time_causal( 0, 0.5, 1, x, physical_values.hertz( 4e6 ) )	% absorption model for the lossy homogeneous fluid
        method_faces ( 1, 1 ) scattering.sequences.setups.discretizations.methods.method { mustBeNonempty } = scattering.sequences.setups.discretizations.methods.grid_numbers( [ 4; 53 ] ); % discretization parameters for the transducer array
        index_selected_tx_ref ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 64	% tx element for reference pulse-echo response
        index_selected_rx_ref ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 64	% rx element for reference pulse-echo response

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = PER( durations_window_t, indices_elements_tx, indices_elements_rx, intervals_f, handles_absorption_model, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class physical_values.time for durations_window_t

            % superclass ensures cell array of positive integers for indices_elements_tx

            % superclass ensures cell array of positive integers for indices_elements_rx

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

            % property validation function ensures class function_handle for handles_absorption_model

            % ensure nonempty methods_faces
            if nargin >= 6 && ~isempty( varargin{ 1 } )
                methods_faces = varargin{ 1 };
            else
                methods_faces = repmat( scattering.sequences.setups.discretizations.methods.grid_numbers( [ 4; 53 ] ), size( durations_window_t ) );
            end

            % property validation function ensures scattering.sequences.setups.discretizations.methods.method for methods_faces

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( durations_window_t, intervals_f, handles_absorption_model, methods_faces );

            %--------------------------------------------------------------
            % 2.) create calibration options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@calibration.options.common( durations_window_t, indices_elements_tx, indices_elements_rx, varargin{ : } );

            % iterate calibration options
            for index_object = 1:numel( objects )

                % ensure class function_handle
                if ~isa( handles_absorption_model{ index_object }, 'function_handle' )
                    errorStruct.message = sprintf( 'handles_absorption_model{ %d } must be function_handle!', index_object );
                    errorStruct.identifier = 'estimate_PER_point:NoFunctionHandles';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).interval_f = intervals_f( index_object );
                objects( index_object ).handle_absorption_model = handles_absorption_model{ index_object };
                objects( index_object ).method_faces = methods_faces( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = PER( durations_window_t, indices_elements_tx, indices_elements_rx, intervals_f, handles_absorption_model, varargin )

	end % methods

end % classdef PER < calibration.options.common
