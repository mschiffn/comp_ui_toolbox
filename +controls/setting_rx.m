%
% superclass for all transducer control settings in recording mode
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-04-04
%
classdef setting_rx < controls.setting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        interval_t ( 1, 1 ) math.interval	% recording time interval
        interval_f ( 1, 1 ) math.interval	% frequency interval

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rx( indices_active, impulse_responses, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            if nargin == 0
                indices_active = 1;
                impulse_responses = physical_values.impulse_response( discretizations.set_discrete_time_regular( 0, 0, physical_values.time(1) ), physical_values.physical_value(1) );
                intervals_t = math.interval( physical_values.time( 0 ), physical_values.time( 1 ) );
                intervals_f = math.interval( physical_values.frequency( 1 ), physical_values.frequency( 2 ) );
            end

            objects@controls.setting( indices_active, impulse_responses );

            %--------------------------------------------------------------
            % 2.) check arguments
            %--------------------------------------------------------------
            % ensure classes math.interval
            if ~( isa( intervals_t, 'math.interval' ) && isa( intervals_f, 'math.interval' ) )
                errorStruct.message     = 'intervals_t and intervals_f must be math.interval!';
                errorStruct.identifier	= 'setting_rx:NoIntervals';
                error( errorStruct );
            end

            % multiple objects / single intervals_t
            if ~isscalar( objects ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( objects ) );
            end

            % multiple objects / single intervals_f
            if ~isscalar( objects ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( objects ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 3.) create recording settings
            %--------------------------------------------------------------
            % iterate transducer control settings in recording mode
            for index_object = 1:numel( objects )

                % ensure time interval
                if ~isa( intervals_t( index_object ).lb, 'physical_values.time' )
                    errorStruct.message = sprintf( 'Bounds of intervals_t( %d ) must be physical_values.time!', index_object );
                    errorStruct.identifier = 'setting_rx:NoTimeInterval';
                    error( errorStruct );
                end

                % ensure frequency interval
                if ~isa( intervals_f( index_object ).lb, 'physical_values.frequency' )
                    errorStruct.message = sprintf( 'Bounds of intervals_f( %d ) must be physical_values.frequency!', index_object );
                    errorStruct.identifier = 'setting_rx:NoFrequencyInterval';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).interval_t = intervals_t( index_object );
                objects( index_object ).interval_f = intervals_f( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = setting_rx( indices_active, impulse_responses, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function settings_rx = discretize( settings_rx, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of arguments
            if ~( nargin == 1 || nargin == 3 )
                errorStruct.message     = 'Either one or three arguments are required!';
                errorStruct.identifier	= 'discretize:Arguments';
                error( errorStruct );
            end

            % specify recording time and frequency intervals
            if nargin == 1

                % use intervals from transducer control settings
                intervals_t = reshape( [ settings_rx.interval_t ], size( settings_rx ) );
                intervals_f = reshape( [ settings_rx.interval_f ], size( settings_rx ) );

            else

                % use external intervals
                intervals_t = varargin{ 1 };
                intervals_f = varargin{ 2 };

            end % if nargin == 1

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples via superclass method
            %--------------------------------------------------------------
            settings_rx = discretize@controls.setting( settings_rx, intervals_t, intervals_f );

        end % function settings_rx = discretize( settings_rx, varargin )

	end % methods

end % classdef setting_rx < controls.setting
