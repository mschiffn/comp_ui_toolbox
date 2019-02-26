%
% superclass for all mixer outputs
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-02-26
%
classdef setting_rx < controls.setting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        interval_t ( 1, 1 ) physical_values.interval_time       % recording time interval
        interval_f ( 1, 1 ) physical_values.interval_frequency	% frequency interval

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
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
                intervals_t = physical_values.interval_time( physical_values.time( 0 ), physical_values.time( 1 ) );
                intervals_f = physical_values.interval_frequency( physical_values.frequency( 1 ), physical_values.frequency( 2 ) );
            end

            objects@controls.setting( indices_active, impulse_responses );

            %--------------------------------------------------------------
            % 2.) check arguments
            %--------------------------------------------------------------
            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 3.) create mixer outputs
            %--------------------------------------------------------------
            % set independent properties
            for index_object = 1:numel( objects )

                objects( index_object ).interval_t = intervals_t( index_object );
                objects( index_object ).interval_f = intervals_f( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = setting_rx( indices_active, impulse_responses, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function transfer_functions = discretize( settings_rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            transfer_functions = cell( size( settings_rx ) );
            for index_object = 1:numel( settings_rx )

                transfer_functions{ index_object } = fourier_transform( settings_rx( index_object ).impulse_responses, settings_rx( index_object ).interval_t, settings_rx( index_object ).interval_f );
            end

        end % function transfer_functions = discretize( settings_rx )

	end % methods

end % classdef setting_rx
