%
% superclass for all time intervals
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-02-01
%
classdef time_interval < physical_values.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = time_interval( times )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'time_interval:Arguments';
                error( errorStruct );
            end

            % prevent emptyness of the argument
            mustBeNonempty( times );

            % arguments must be physical_values.time
            if ~isa( times, 'physical_values.time' )
                errorStruct.message     = 'times must be physical_values.time!';
                errorStruct.identifier	= 'time_interval:Arguments';
                error( errorStruct );
            end

            % constructor of superclass
            obj@physical_values.interval( times );
        end

	end % methods

end % classdef time_interval
