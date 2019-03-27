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
% modified: 2019-03-26
%
classdef transparent_container

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = protected)

        % independent properties
        values

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
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
        % 1.) overload built-in functions to ensure transparency
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
        % redefine subscripted reference for objects
        function container = subsref( container, S )
            container.values = subsref( container.values, S );
        end
        % redefine subscripted assignment
%         function A = subsasgn(A,S,B)
%         end
        % display value of variable
        function disp( container )
            disp( container.values );
        end

	end % methods

end % classdef transparent_container
