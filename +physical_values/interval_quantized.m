%
% superclass for all quantized intervals of physical values
%
% author: Martin F. Schiffner
% date: 2019-02-06
% modified: 2019-02-14
%
classdef interval_quantized < physical_values.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q_lb ( 1, 1 ) int64     % lower integer bound
        q_ub ( 1, 1 ) int64     % upper integer bound
        delta ( 1, 1 ) physical_values.physical_value       % quantization step

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = interval_quantized( lbs_q, ubs_q, deltas )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integers
            if ~( all( lbs_q( : ) == floor( lbs_q( : ) ) ) && all( ubs_q( : ) == floor( ubs_q( : ) ) ) )
                errorStruct.message     = 'Boundary indices must be integers!';
                errorStruct.identifier	= 'interval:NoIntegers';
                error( errorStruct );
            end

            % ensure class physical_values.physical_value
            if ~isa( deltas, 'physical_values.physical_value' )
                errorStruct.message     = 'deltas must be physical_values.physical_value!';
                errorStruct.identifier	= 'interval:NoPhysicalValues';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs_q, ubs_q, deltas );
            % assertion: lbs_q, ubs_q, and deltas have equal sizes

            %--------------------------------------------------------------
            % 2.) compute lower and upper bounds
            %--------------------------------------------------------------
            lbs = lbs_q .* deltas;
            ubs = ubs_q .* deltas;

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.interval( lbs, ubs );
            % assertions: lbs and ubs are nonempty, have equal size, are physical_values.physical_value, and increase strictly monotonic

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            N_objects = numel( objects );
            for index_object = 1:N_objects
                objects( index_object ).q_lb = int64( lbs_q( index_object ) );
                objects( index_object ).q_ub = int64( ubs_q( index_object ) );
                objects( index_object ).delta = deltas( index_object );
            end

        end

	end % methods

end % classdef interval_quantized
