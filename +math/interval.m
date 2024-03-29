%
% superclass for all intervals of physical quantities
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2020-04-13
%
classdef interval

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        lb ( 1, 1 ) physical_values.physical_quantity	% lower bound
        ub ( 1, 1 ) physical_values.physical_quantity	% upper bound

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
            if ~all( lbs < ubs, 'all' )
                errorStruct.message = 'Interval bounds must increase strictly monotonic!';
                errorStruct.identifier = 'interval:NoStrictIncrease';
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
        % move
        %------------------------------------------------------------------
        function intervals = move( intervals, centers )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.interval
            if ~isa( intervals, 'math.interval' )
                errorStruct.message = 'intervals must be math.interval!';
                errorStruct.identifier = 'move:NoIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', intervals.lb, centers );

            % single intervals / multiple centers
            if isscalar( intervals ) && ~isscalar( centers )
                intervals = repmat( intervals, size( centers ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( intervals, centers );

            %--------------------------------------------------------------
            % 2.) move intervals
            %--------------------------------------------------------------
            % compute deltas
            deltas = centers - center( intervals );

            % update lower and upper bounds
            lbs = reshape( [ intervals.lb ], size( intervals ) ) + deltas;
            ubs = reshape( [ intervals.ub ], size( intervals ) ) + deltas;

            % iterate intervals
            for index_interval = 1:numel( intervals )

                intervals( index_interval ).lb = lbs( index_interval );
                intervals( index_interval ).ub = ubs( index_interval );

            end % for index_interval = 1:numel( intervals )

        end % function intervals = move( intervals, centers )

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
        % quantization
        %------------------------------------------------------------------
        function objects_out = quantize( intervals, deltas, expand )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure at least two and at most three arguments
            narginchk( 2, 3 );

            % ensure class math.interval
            if ~isa( intervals, 'math.interval' )
                errorStruct.message = 'intervals must be math.interval!';
                errorStruct.identifier = 'quantize:NoIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', intervals.lb, deltas );

            % ensure nonempty expand
            if nargin < 3 || isempty( expand )
                expand = false;
            end

            % ensure logical for expand
            if ~islogical( expand )
                errorStruct.message = 'expand must be logical!';
                errorStruct.identifier = 'quantize:NoLogicals';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ intervals, deltas, expand ] = auxiliary.ensureEqualSize( intervals, deltas, expand );

            %--------------------------------------------------------------
            % 2.) compute boundary indices
            %--------------------------------------------------------------
            % extract upper bounds
            lbs = reshape( [ intervals.lb ], size( intervals ) );
            ubs = reshape( [ intervals.ub ], size( intervals ) );

            % compute lower and upper boundary indices
            lbs_exact = lbs ./ deltas;
            lbs_q = ceil( lbs_exact );
            lbs_q( expand ) = floor( lbs_exact( expand ) );

            ubs_exact = ubs ./ deltas;
            ubs_q = floor( ubs_exact );
            ubs_q( expand ) = ceil( ubs_exact( expand ) );

            % correct rounding errors
            lbs_rnd = round( lbs_exact );
            indicator_rnd = abs( lbs_exact - lbs_rnd ) <= eps( lbs_rnd );
            lbs_q( indicator_rnd ) = lbs_rnd( indicator_rnd );

            ubs_rnd = round( ubs_exact );
            indicator_rnd = abs( ubs_exact - ubs_rnd ) <= eps( ubs_rnd );
            ubs_q( indicator_rnd ) = ubs_rnd( indicator_rnd );

            %--------------------------------------------------------------
            % 3.) create quantized intervals
            %--------------------------------------------------------------
            objects_out = math.interval_quantized( lbs_q, ubs_q, deltas );

        end % function objects_out = quantize( intervals, deltas, expand )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function objects_out = discretize( intervals, deltas )
% TODO: options discretization method

            %--------------------------------------------------------------
            % 1.) quantize intervals
            %--------------------------------------------------------------
            % ensure quantized intervals
% TODO: wrong! -> quantization with same deltas!!!
            if ~isa( intervals, 'math.interval_quantized' )
                intervals = quantize( intervals, deltas );
            end

            %--------------------------------------------------------------
            % 2.) extract quantized lower and upper bounds
            %--------------------------------------------------------------
            lbs_q = reshape( [ intervals.q_lb ], size( intervals ) );
            ubs_q = reshape( [ intervals.q_ub ], size( intervals ) );

            %--------------------------------------------------------------
            % 3.) create strictly monotonically increasing sequences
            %--------------------------------------------------------------
            objects_out = math.sequence_increasing_regular_quantized( lbs_q, ubs_q, deltas );

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
