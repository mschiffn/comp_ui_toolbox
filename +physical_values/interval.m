%
% superclass for all intervals of physical values
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-02-12
%
classdef interval < physical_values.physical_quantity

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        lb physical_values.physical_quantity	% lower bound
        ub physical_values.physical_quantity	% upper bound

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
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

            % ensure equal subclasses of physical_values.physical_value
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_value', lbs, ubs );

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
            % construct objects
            N_objects = numel( lbs );
            objects = repmat( objects, size( lbs ) );

            % set independent properties
            for index_object = 1:N_objects

                objects( index_object ).lb = lbs( index_object );
                objects( index_object ).ub = ubs( index_object );
            end

        end % function objects = interval( lbs, ubs )

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        function objects_out = center( objects_in )

            % compute centers
            lbs = [ objects_in.lb ];
            ubs = [ objects_in.ub ];
            objects_out = reshape( lbs + ubs, size( objects_in ) ) ./ 2;

        end % function objects_out = center( objects_in )

        %------------------------------------------------------------------
        % length (overload abs function)
        %------------------------------------------------------------------
        function lengths = abs( objects_in )

            % compute lengths
            lbs = [ objects_in.lb ];
            ubs = [ objects_in.ub ];
            lengths = reshape( ubs - lbs, size( objects_in) );

        end % function lengths = abs( objects_in )

        %------------------------------------------------------------------
        % element
        %------------------------------------------------------------------
        function indicator = element( objects, values )

            % extract lower and upper bounds
            lbs = [ objects.lb ];
            ubs = [ objects.ub ];

            % TODO: check sizes of objects vs values

            % ensure equal subclasses of physical_values.physical_value
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_value', lbs, values );

            % compute results
            indicator = ( lbs <= values ) && ( values <= ubs );

        end % function indicator = element( objects, values )

        %------------------------------------------------------------------
        % quantization (overload quantize function)
        %------------------------------------------------------------------
        function objects_out = quantize( objects_in, deltas )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_in, deltas );
            % assertion: objects_in and deltas have equal size

            % ensure equal subclasses of physical_values.physical_value
            lbs = [ objects_in.lb ];
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_value', lbs, deltas );
            % assertion: objects_in.lb, objects_in.ub, and deltas are equal subclasses of physical_values.physical_value

            %--------------------------------------------------------------
            % 2.) compute boundary indices
            %--------------------------------------------------------------
            % initialize boundary indices
            lbs_q = zeros( size( objects_in ) );
            ubs_q = zeros( size( objects_in ) );

            % compute lower and upper boundary indices
            for index_object = 1:numel( objects_in )

                % compute boundary fractions
                index_lb = objects_in( index_object ).lb ./ deltas( index_object );
                index_ub = objects_in( index_object ).ub ./ deltas( index_object );

                % compute boundary indices
                lbs_q( index_object ) = ceil( index_lb );
                ubs_q( index_object ) = floor( index_ub );

            end

            %--------------------------------------------------------------
            % 3.) construct quantized intervals
            %--------------------------------------------------------------
            objects_out = physical_values.interval_quantized( lbs_q, ubs_q, deltas );

        end % function objects_out = quantize( objects_in, deltas )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function objects_out = discretize( objects_in, deltas )
            % TODO: enumeration class discretization method

            %--------------------------------------------------------------
            % 1.) quantize intervals
            %--------------------------------------------------------------
            objects_in = quantize( objects_in, deltas );
            % assertion: objects_in is physical_values.interval_quantized

            %--------------------------------------------------------------
            % 2.) compute sets of discrete physical values
            %--------------------------------------------------------------
            % create cell array of objects
            sets = cell( size( objects_in ) );

            % set independent properties
            for index_object = 1:numel( objects_in )

                % compute discrete times
                sets{ index_object } = double( objects_in( index_object ).q_lb:(objects_in( index_object ).q_ub - 1) ) .* objects_in( index_object ).delta;
            end

            %--------------------------------------------------------------
            % 3.) construct sets of discrete physical values
            %--------------------------------------------------------------
            objects_out = discretizations.set_discrete_physical_value( sets );

        end % function objects_out = discretize( objects_in, deltas )

        %------------------------------------------------------------------
        % convex hull of intervals
        %------------------------------------------------------------------
        function object_out = hull( objects_in )

            % determine minimum lower bound and maximum upper bound
            lb_min = min( [ objects_in.lb ] );
            ub_max = max( [ objects_in.ub ] );

            % set independent properties
            object_out = objects_in( 1 );
            object_out.lb = lb_min;
            object_out.ub = ub_max;

        end % function object_out = hull( objects_in )

	end % methods

end % classdef interval
