%
% superclass for all sets of discrete physical values
%
% author: Martin F. Schiffner
% date: 2019-02-07
% modified: 2019-02-07
%
classdef set_discrete_physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        S ( 1, : ) physical_values.physical_value	% set consists of multiple physical values

        % dependent properties
        N_t ( 1, 1 ) double = 0                     % number of discrete physical values

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_physical_value( intervals, deltas )

            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class physical_values.interval
            if ~isa( intervals, 'physical_values.interval' )
                errorStruct.message     = 'intervals must be physical_values.interval!';
                errorStruct.identifier	= 'set_discrete_physical_value:NoInterval';
                error( errorStruct );
            end
            % assertion: intervals is physical_values.interval

            % ensure quantized intervals
            if ~isa( intervals, 'physical_values.interval_quantized' )
                intervals = quantize( intervals, deltas );
            end
            % assertion: intervals is physical_values.interval_quantized

            %--------------------------------------------------------------
            % 2.) compute sets of discrete physical values
            %--------------------------------------------------------------
            % create column vector of objects
            N_objects = numel( intervals );
            objects = repmat( objects, size( intervals ) );

            % set independent properties
            for index_object = 1:N_objects

                % compute discrete times
                objects( index_object ).S = double( intervals( index_object ).bounds_q( 1 ):intervals( index_object ).bounds_q( 2 ) ) .* intervals( index_object ).delta;

                % set dependent properties
                objects( index_object ).N_t = numel( objects( index_object ).S );

            end

        end % function objects = set_discrete_physical_value( interval, f_s )

	end % methods

end % classdef set_discrete_physical_value
