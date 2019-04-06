%
% superclass for all transparent containers
%
% inspired by:
%  [1] physical-units-for-matlab
%      www.mathworks.com/matlabcentral/fileexchange/authors/101715
%      github.com/sky-s/physical-units-for-matlab
%
% author: Martin F. Schiffner
% date: 2019-03-25
% modified: 2019-04-04
%
classdef transparent_container

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = protected)

        % independent properties
        values

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        %% constructor
        %------------------------------------------------------------------
        function object = transparent_container( values )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            object.values = values;

        end % function object = transparent_container( values )

        %------------------------------------------------------------------
        %% 1.) overload built-in functions to ensure transparency
        %------------------------------------------------------------------
        % number of array elements
        function result = numel( container )
            result = numel( container.values );
        end

        % array size
        function varargout = size( container, varargin )
            [ varargout{ 1:nargout } ] = size( container.values, varargin{ : } );
        end

        % length of largest array dimension
        function result = length( container )
            result = length( container.values );
        end

        % number of array dimensions
        function result = ndims( container )
            result = ndims( container.values );
        end

        % repeat copies of array
        function container = repmat( container, varargin )
            container.values = repmat( container.values, varargin{ : } );
        end

        % reshape array
        function container = reshape( container, varargin )
            container.values = reshape( container.values, varargin{ : } );
        end

        % shift array circularly
        function container = circshift( container, varargin )
            container.values = circshift( container.values, varargin{ : } );
        end
    
        % redefine subscripted reference for objects
        function container = subsref( container, S )
            container.values = subsref( container.values, S );
        end

        % redefine subscripted assignment
        function container = subsasgn( container, S, B )
            
            if isempty( container )
                % TODO: is this correct?
                container = B;
            else
                container.values = subsasgn( container.values, S, B.values );
            end

        end

        % indicate last array index
        function index_end = end( container, index_end, N_indices )

            % get size of content
            size_act = size( container.values );

            % return index
            if N_indices == 1
                index_end = prod( size_act );
            else
                index_end = size_act( index_end );
            end
        end

        % display value of variable
        function disp( container )
            if ~isempty( container )
                disp( container.values );
            end
        end

        %------------------------------------------------------------------
        %% 2.) overload built-in state detection (is*) functions
        %------------------------------------------------------------------
        % see Matlab documentation "is*" "Detect state"
        % determine if input is cell array
        function tf = iscell( container )
            tf = iscell( container.values );
        end

        % determine if input is cell array of character vectors
        function tf = iscellstr( container )
            tf = iscellstr( container.values );
        end

        % determine if input is character array
        function tf = ischar( container )
            tf = ischar( container.values );
        end

        % determine whether input is column vector
        function tf = iscolumn( container )
            tf = iscolumn( container.values );
        end

        % determine if matrix is diagonal
        function tf = isdiag( container )
            tf = isdiag( container.values );
        end

        % determine whether array is empty
        function tf = isempty( container )
            tf = isempty( container.values );
        end

        % determine whether array is real
        function tf = isreal( container )
            tf = isreal( container.values );
        end

        % determine if input is numeric array
        function tf = isnumeric( container )
            tf = isnumeric( container.values );
        end

        % array elements that are finite
        function tf = isfinite( container )
            tf = isfinite( container.values );
        end

        % determine if input is floating-point array
        function tf = isfloat( container )
            tf = isfloat( container.values );
        end

        % array elements that are infinite
        function tf = isinf( container )
            tf = isinf( container.values );
        end

        % array elements that are NaN
        function tf = isnan( container )
            tf = isnan( container.values );
        end

        % determine if input is logical array
        function tf = islogical( container )
            tf = islogical( container.values );
        end

        % determine if array is sorted
        function result = issorted( container, varargin )
            result = issorted( container.values, varargin{ : } );
        end

        %------------------------------------------------------------------
        %% 3.) overload built-in property validation (mustBe*) functions
        %------------------------------------------------------------------
        % validate that value is positive or issue error
        function mustBePositive( container )
            mustBePositive( container.values );
        end

        % validate that value is nonpositive or issue error
        function mustBeNonpositive( container )
            mustBeNonpositive( container.values );
        end

        % validate that value is finite or issue error
        function mustBeFinite( container )
            mustBeFinite( container.values );
        end

        % validate that value is nonNaN
        function mustBeNonNan( container )
            mustBeNonNan( container.values );
        end

        % validate that value is negative or issue error
        function mustBeNegative( container )
            mustBeNegative( container.values );
        end

        % validate that value is nonnegative or issue error
        function mustBeNonnegative( container )
            mustBeNonnegative( container.values );
        end

        % validate that value is nonzero or issue error
        function mustBeNonzero( container )
            mustBeNonzero( container.values );
        end

        % validate that value is greater than another value or issue error
        function mustBeGreaterThan( container_1, container_2 )
            mustBeGreaterThan( container_1.values, container_2.values );
        end

        % validate that value is greater than or equal to another value or issue error
        function mustBeGreaterThanOrEqual( container_1, container_2 )
            mustBeGreaterThanOrEqual( container_1.values, container_2.values );
        end

        % validate that value is less than another value or issue error
        function mustBeLessThan( container_1, container_2 )
            mustBeLessThan( container_1.values, container_2.values );
        end

        % validate that value is less than or equal to another value or issue error
        function mustBeLessThanOrEqual( container_1, container_2 )
            mustBeLessThanOrEqual( container_1.values, container_2.values );
        end

        % validate that value is nonempty or issue error
        function mustBeNonempty( container )
            mustBeNonempty( container.values );
        end

        % validate that value is nonsparse or issue error
        function mustBeNonsparse( container )
            mustBeNonsparse( container.values );
        end

        % validate that value is numeric or issue error
        function mustBeNumeric( container )
            mustBeNumeric( container.values );
        end

        % validate that value is numeric or logical or issue error
        function mustBeNumericOrLogical( container )
            mustBeNumericOrLogical( container.values );
        end

        % validate that value is real or issue error
        function mustBeReal( container )
            mustBeReal( container.values );
        end

        % validate that value is integer or issue error
        function mustBeInteger( container )
            mustBeInteger( container.values );
        end

        % validate that value is member of specified set
%         mustBeMember

	end % methods

end % classdef transparent_container
