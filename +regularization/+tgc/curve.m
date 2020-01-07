%
% abstract superclass for all time gain compensation curves
%
% author: Martin F. Schiffner
% date: 2019-12-07
% modified: 2019-12-23
%
classdef (Abstract) curve

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
% TODO: function handle
        % independent properties
        interval_t ( 1, 1 ) math.interval { mustBeNonempty } = math.interval( physical_values.second( 0 ), physical_values.second( 1e-6 ) ) % recording time interval

        % dependent properties
        T ( 1, 1 ) physical_values.time { mustBePositive, mustBeNonempty } = physical_values.second( 1e-6 )

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = curve( intervals_t )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.interval
            if ~isa( intervals_t, 'math.interval' )
                errorStruct.message = 'intervals_t must be math.interval!';
                errorStruct.identifier = 'exponential:NoIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.time
            auxiliary.mustBeEqualSubclasses( 'physical_values.time', intervals_t.lb );

            %--------------------------------------------------------------
            % 2.) create time gain compensation curves
            %--------------------------------------------------------------
            % repeat default time gain compensation curves
            objects = repmat( objects, size( intervals_t ) );

            % iterate time gain compensation curves
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).interval_t = intervals_t( index_object );

                % set dependent properties
                objects( index_object ).T = abs( objects( index_object ).interval_t );

            end % for index_object = 1:numel( objects )

        end % function objects = curve( intervals_t )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % sample time gain compensation curves
        %------------------------------------------------------------------
        samples = sample_curve( curves, axes )

        %------------------------------------------------------------------
        % Fourier coefficients
        %------------------------------------------------------------------
        signal_matrices = fourier_coefficients( curves, varargin )

    end % methods (Abstract)

end % classdef (Abstract) curve
