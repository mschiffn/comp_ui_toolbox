%
% superclass for all intervals of physical quantities
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-05-06
%
classdef interval

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        lb physical_values.physical_quantity	% lower bound
        ub physical_values.physical_quantity	% upper bound

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = interval( lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return empty object if no arguments
            if nargin == 0
                return;
            end

            % prevent empty arguments
            mustBeNonempty( lbs );
            mustBeNonempty( ubs );

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', lbs, ubs );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs, ubs );

            % ensure strictly monotonic increasing bounds
            if ~all( ubs > lbs, 'all' )
                errorStruct.message     = 'Interval bounds must increase strictly monotonic!';
                errorStruct.identifier	= 'interval:NoStrictIncrease';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create intervals
            %--------------------------------------------------------------
            % repeat objects
            objects = repmat( objects, size( lbs ) );

            % set independent properties
            for index_object = 1:numel( lbs )

                objects( index_object ).lb = lbs( index_object );
                objects( index_object ).ub = ubs( index_object );
            end

        end % function objects = interval( lbs, ubs )

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        function objects_out = center( intervals )

            % compute centers
            lbs = [ intervals.lb ];
            ubs = [ intervals.ub ];
            objects_out = reshape( lbs + ubs, size( intervals ) ) ./ 2;

        end % function objects_out = center( intervals )

        %------------------------------------------------------------------
        % length (overload abs function)
        %------------------------------------------------------------------
        function lengths = abs( intervals )

            % compute lengths
            lbs = [ intervals.lb ];
            ubs = [ intervals.ub ];
            lengths = reshape( ubs - lbs, size( intervals) );

        end % function lengths = abs( intervals )

        %------------------------------------------------------------------
        % element
        %------------------------------------------------------------------
        function indicator = element( objects, values )

            % extract lower and upper bounds
            lbs = [ objects.lb ];
            ubs = [ objects.ub ];

            % TODO: check sizes of objects vs values

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', lbs, values );

            % compute results
            indicator = ( lbs <= values ) && ( values <= ubs );

        end % function indicator = element( objects, values )

        %------------------------------------------------------------------
        % quantization (overload quantize function)
        %------------------------------------------------------------------
        function objects_out = quantize( intervals, deltas )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( intervals, deltas );

            % ensure equal subclasses of physical_values.physical_quantity
            lbs = [ intervals.lb ];
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', lbs, deltas );

            %--------------------------------------------------------------
            % 2.) compute boundary indices
            %--------------------------------------------------------------
            % initialize boundary indices
            lbs_q = zeros( size( intervals ) );
            ubs_q = zeros( size( intervals ) );

            % compute lower and upper boundary indices
            for index_object = 1:numel( intervals )

                % compute boundary fractions
                index_lb = intervals( index_object ).lb / deltas( index_object );
                index_ub = intervals( index_object ).ub / deltas( index_object );

                % compute boundary indices
                lbs_q( index_object ) = ceil( index_lb );
                ubs_q( index_object ) = floor( index_ub );

            end % for index_object = 1:numel( intervals )

            %--------------------------------------------------------------
            % 3.) create quantized intervals
            %--------------------------------------------------------------
            objects_out = math.interval_quantized( lbs_q, ubs_q, deltas );

        end % function objects_out = quantize( intervals, deltas )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function objects_out = discretize( intervals, deltas )
            % TODO: enumeration class discretization method

            %--------------------------------------------------------------
            % 1.) quantize intervals
            %--------------------------------------------------------------
            intervals = quantize( intervals, deltas );

            %--------------------------------------------------------------
            % 2.) extract quantized lower and upper bounds
            %--------------------------------------------------------------
            lbs_q = reshape( [ intervals.q_lb ], size( intervals ) );
            ubs_q = reshape( [ intervals.q_ub ], size( intervals ) );

            %--------------------------------------------------------------
            % 3.) create strictly monotonically increasing sequences
            %--------------------------------------------------------------
            objects_out = math.sequence_increasing_regular( lbs_q, ubs_q, deltas );

        end % function objects_out = discretize( intervals, deltas )

        %------------------------------------------------------------------
        % convex hull of intervals
        %------------------------------------------------------------------
        function interval_out = hull( intervals )

            % determine minimum lower bound and maximum upper bound
            lb_min = min( [ intervals.lb ] );
            ub_max = max( [ intervals.ub ] );

            % create interval
            interval_out = math.interval( lb_min, ub_max );

        end % function interval_out = hull( intervals )

	end % methods

end % classdef interval
