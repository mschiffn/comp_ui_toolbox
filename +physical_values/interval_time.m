%
% superclass for all time intervals
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-02-14
%
classdef interval_time < physical_values.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = interval_time( lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                lbs = physical_values.time( 0 );
                ubs = physical_values.time( 1 );
            end

            % arguments must be physical_values.time
            if ~( isa( lbs, 'physical_values.time' ) && isa( ubs, 'physical_values.time' ) )
                errorStruct.message     = 'Both arguments must be physical_values.time!';
                errorStruct.identifier	= 'interval_time:NoTimes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.interval( lbs, ubs );

        end % function objects = interval_time( lbs, ubs )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function objects_out = discretize( objects_in, deltas )
            % TODO: enumeration class discretization method: regular sampling vs random sampling

            %--------------------------------------------------------------
            % 1.) quantize intervals
            %--------------------------------------------------------------
            objects_in = quantize( objects_in, deltas );
            % assertion: objects_in is physical_values.interval_quantized

            %--------------------------------------------------------------
            % 2.) create regular sets of discrete times
            %--------------------------------------------------------------
            objects_out = discretizations.set_discrete_time_regular( double( [ objects_in.q_lb ] ), double( [ objects_in.q_ub ] - 1 ), deltas );

        end % function objects_out = discretize( objects_in, deltas )

	end % methods

end % classdef interval_time
