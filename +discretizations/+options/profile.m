%
% superclass for all projected profile options
%
% author: Martin F. Schiffner
% date: 2020-01-08
% modified: 2020-01-08
%
classdef profile

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        dim ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1   % projection dimension
        interval ( 1, 1 ) math.interval { mustBeNonempty } = math.interval          % projection interval
        N_zeros_add ( 1, 1 ) double { mustBeNonnegative, mustBeInteger, mustBeNonempty } = 50	% dimension
        factor_interp ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 10	% dimension

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = profile( dims, intervals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid dims
            % ensure class math.interval
            if ~isa( intervals, 'math.interval' )
                errorStruct.message = 'intervals must be math.interval!';
                errorStruct.identifier = 'profile:NoIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.length
            auxiliary.mustBeEqualSubclasses( 'physical_values.length', intervals.lb );

            % multiple dims / single intervals
            if ~isscalar( dims ) && isscalar( intervals )
                intervals = repmat( intervals, size( dims ) );
            end

            % single dims / multiple intervals
            if isscalar( dims ) && ~isscalar( intervals )
                dims = repmat( dims, size( intervals ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( dims, intervals );

            %--------------------------------------------------------------
            % 2.) create profile options
            %--------------------------------------------------------------
            % repeat default profile options
            objects = repmat( objects, size( dims ) );

            % iterate profile options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).dim = dims( index_object );
                objects( index_object ).interval = intervals( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = profile( dims, intervals )

	end % methods

end % classdef profile
