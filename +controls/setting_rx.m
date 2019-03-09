%
% superclass for all recording settings
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-03-08
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
            % 3.) create recording settings
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
        function objects_out = discretize( settings_rx, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if ~( nargin == 1 || nargin == 3 )
                errorStruct.message     = 'Either one or three arguments are required!';
                errorStruct.identifier	= 'discretize:Arguments';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute transfer functions
            %--------------------------------------------------------------
            transfer_functions = cell( size( settings_rx ) );
            for index_object = 1:numel( settings_rx )

                if nargin == 1
                    interval_t_act = settings_rx( index_object ).interval_t;
                    interval_f_act = settings_rx( index_object ).interval_f;
                else
                    interval_t_act = varargin{ 1 };
                    interval_f_act = varargin{ 2 };
                end

                transfer_functions{ index_object } = fourier_transform( settings_rx( index_object ).impulse_responses, interval_t_act, interval_f_act );

            end % for index_object = 1:numel( settings_rx )

            % create spectral discretizations of the recording settings
            objects_out = discretizations.spectral_points_rx( transfer_functions );

        end % function objects_out = discretize( settings_rx, varargin )

	end % methods

end % classdef setting_rx < controls.setting
