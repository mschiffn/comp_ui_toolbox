%
% superclass for all frequency intervals
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-01-21
%
classdef frequency_interval < physical_values.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = frequency_interval( frequencies )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'frequency_interval:Arguments';
                error( errorStruct );
            end

            % prevent emptyness of the argument
            mustBeNonempty( frequencies );

            % arguments must be physical_values.time
            if ~isa( frequencies, 'physical_values.frequency' )
                errorStruct.message     = 'frequencies must be physical_values.frequency!';
                errorStruct.identifier	= 'frequency_interval:Arguments';
                error( errorStruct );
            end

            % constructor of superclass
            obj@physical_values.interval( frequencies );
        end
	end % methods

end % classdef frequency_interval
