%
% superclass for all frequency intervals
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-02-20
%
classdef interval_frequency < physical_values.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = interval_frequency( lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                lbs = physical_values.frequency( 1 );
                ubs = physical_values.frequency( 2 );
            end

            % arguments must be physical_values.frequency
            if ~( isa( lbs, 'physical_values.frequency' ) && isa( ubs, 'physical_values.frequency' ) )
                errorStruct.message     = 'Both arguments must be physical_values.time!';
                errorStruct.identifier	= 'interval_time:NoTimes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.interval( lbs, ubs );

        end % function objects = interval_frequency( lbs, ubs )

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
            % 2.) create regular sets of discrete frequencies
            %--------------------------------------------------------------
            objects_out = discretizations.set_discrete_frequency_regular( double( [ objects_in.q_lb ] ), double( [ objects_in.q_ub ] ), deltas );

        end % function objects_out = discretize( objects_in, deltas )

	end % methods

end % classdef interval_frequency
