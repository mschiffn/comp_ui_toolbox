%
% superclass for all quantized time intervals
%
% author: Martin F. Schiffner
% date: 2019-02-06
% modified: 2019-03-28
%
classdef interval_quantized_time < math.interval_quantized

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = interval_quantized_time( lbs, ubs, deltas )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % arguments must be physical_values.time
            if ~isa( deltas, 'physical_values.time' )
                errorStruct.message     = 'deltas must be physical_values.time!';
                errorStruct.identifier	= 'interval_quantized_time:NoTime';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@math.interval_quantized( lbs, ubs, deltas );

        end

	end % methods

end % classdef interval_quantized_time
