%
% superclass for all length intervals
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-02-14
%
classdef interval_position < physical_values.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = interval_position( lbs, ubs )

            % set default values
            if nargin == 0
                lbs = physical_values.position( 0 );
                ubs = physical_values.position( 1 );
            end

            % arguments must be physical_values.position
            if ~( isa( lbs, 'physical_values.position' ) && isa( ubs, 'physical_values.position' ) )
                errorStruct.message     = 'Both arguments must be physical_values.position!';
                errorStruct.identifier	= 'interval_position:NoPositions';
                error( errorStruct );
            end

            % constructor of superclass
            objects@physical_values.interval( lbs, ubs );

        end

	end % methods

end % classdef interval_position
