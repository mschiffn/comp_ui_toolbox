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

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                lbs = physical_values.position( 0 );
                ubs = physical_values.position( 1 );
            end

            % arguments must be coordinates.coordinates_cartesian
            if ~( isa( lbs, 'coordinates.coordinates_cartesian' ) && isa( ubs, 'coordinates.coordinates_cartesian' ) )
                errorStruct.message     = 'Both arguments must be coordinates.coordinates_cartesian!';
                errorStruct.identifier	= 'interval_position:NoPositions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.interval( lbs, ubs );

        end % function objects = interval_position( lbs, ubs )

	end % methods

end % classdef interval_position < physical_values.interval
