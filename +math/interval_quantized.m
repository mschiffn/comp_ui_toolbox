%
% superclass for all quantized intervals of physical quantities
%
% author: Martin F. Schiffner
% date: 2019-02-06
% modified: 2019-06-14
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

            % superclass ensures physical_values.physical_quantity for deltas

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs_q, ubs_q, deltas );

            %--------------------------------------------------------------
            % 2.) compute lower and upper bounds
            %--------------------------------------------------------------
            lbs = double( lbs_q ) .* deltas;
            ubs = double( ubs_q ) .* deltas;

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

        %------------------------------------------------------------------
        % quantization (overload quantize method)
        %------------------------------------------------------------------
        function intervals_quantized = quantize( intervals_quantized, deltas )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.interval_quantized
            if ~isa( intervals_quantized, 'math.interval_quantized' )
                errorStruct.message = 'intervals_quantized must be math.interval_quantized!';
                errorStruct.identifier = 'quantize:NoQuantizedInterval';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', intervals_quantized.delta, deltas );

            % multiple intervals_quantized / single deltas
            if ~isscalar( intervals_quantized ) && isscalar( deltas )
                deltas = repmat( deltas, size( intervals_quantized ) );
            end

            % single intervals_quantized / multiple deltas
            if isscalar( intervals_quantized ) && ~isscalar( deltas )
                intervals_quantized = repmat( intervals_quantized, size( deltas ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( intervals_quantized, deltas );

            %--------------------------------------------------------------
            % 2.) requantize quantized intervals
            %--------------------------------------------------------------
            % ensure different quantization steps
            indicator_different = reshape( [ intervals_quantized.delta ], size( intervals_quantized ) ) ~= deltas;

            % call method quantize in superclass
            if sum( indicator_different( : ) ) > 0
                intervals_quantized( indicator_different ) = quantize@math.interval( intervals_quantized( indicator_different ), deltas( indicator_different ) );
            end

        end % function intervals_quantized = quantize( intervals_quantized, deltas )

	end % methods

end % classdef interval_quantized < math.interval
