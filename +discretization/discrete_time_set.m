%
% superclass for all discrete time sets
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-01-16
%
classdef discrete_time_set

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        T ( 1, : ) recordings.time	% set consists of multiple times
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = discrete_time_set( interval_t, f_s )

            if nargin ~= 0

                % create column vector of objects
                N_intervals = size( interval_t, 1 );
                obj( N_intervals, 1 ) = obj;

                % check and set independent properties
                for index_interval = 1:N_intervals

                    % compute lower and upper bounds on the time index
                    q_lb = floor( f_s( index_interval ) * double( interval_t( index_interval ).bounds( 1 ) ) );
                    q_ub = ceil( f_s( index_interval ) * double( interval_t( index_interval ).bounds( 2 ) ) );

                    % compute discrete times
                    obj( index_interval ).T = recordings.time( (q_lb:q_ub) / f_s( index_interval ) );
                end
            end
        end
	end % methods

end % classdef discrete_time_set
