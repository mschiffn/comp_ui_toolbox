%
% superclass for all quantized intervals of physical quantities
%
% author: Martin F. Schiffner
% date: 2019-02-06
% modified: 2019-03-28
%
classdef interval_quantized < math.interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q_lb ( 1, 1 ) int64     % lower integer bound
        q_ub ( 1, 1 ) int64     % upper integer bound
        delta ( 1, 1 ) physical_values.physical_quantity       % quantization step

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
            % ensure integers for lbs_q and ubs_q
            mustBeInteger( lbs_q );
            mustBeInteger( ubs_q );

            % physical_values.physical_value are ensured by superclass

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs_q, ubs_q, deltas );

            %--------------------------------------------------------------
            % 2.) compute lower and upper bounds
            %--------------------------------------------------------------
            lbs = lbs_q .* deltas;
            ubs = ubs_q .* deltas;

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@math.interval( lbs, ubs );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )
                objects( index_object ).q_lb = int64( lbs_q( index_object ) );
                objects( index_object ).q_ub = int64( ubs_q( index_object ) );
                objects( index_object ).delta = deltas( index_object );
            end

        end % function objects = interval_quantized( lbs_q, ubs_q, deltas )

	end % methods

end % classdef interval_quantized < math.interval
