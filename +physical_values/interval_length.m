%
% superclass for all length intervals
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-03-25
%
classdef interval_length < physical_values.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = interval_length( lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                lbs = physical_values.length( 0 );
                ubs = physical_values.length( 1 );
            end

            % arguments must be coordinates.coordinates_cartesian
            if ~( isa( lbs, 'physical_values.length' ) && isa( ubs, 'physical_values.length' ) )
                errorStruct.message     = 'Both arguments must be coordinates.coordinates_cartesian!';
                errorStruct.identifier	= 'interval_length:NoPositions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.interval( lbs, ubs );

        end % function objects = interval_length( lbs, ubs )

	end % methods

end % classdef interval_length < physical_values.interval
