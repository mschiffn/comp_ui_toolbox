%
% superclass for all orthotopes of physical values
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2020-07-02
%
classdef orthotope

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        intervals ( 1, : ) math.interval	% intervals of physical quantities

        % dependent properties
        N_dimensions ( 1, 1 ) { mustBeInteger, mustBeNonempty } = 0     % number of dimensions

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthotope( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                varargin = { math.interval };
            end

            % ensure equal subclasses of math.interval
            auxiliary.mustBeEqualSubclasses( 'math.interval', varargin{ : } );
% TODO: ensure identical units!
            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create orthotopes
            %--------------------------------------------------------------
            % repeat default orthotopes
            objects = repmat( objects, size( varargin{ 1 } ) );

            % iterate orthotopes
            for index_object = 1:numel( varargin{ 1 } )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                % initialize intervals
                objects( index_object ).intervals = repmat( varargin{ 1 }( index_object ), [ numel( varargin ), 1 ] );

                % set intervals
                for index_arg = 2:nargin
                    objects( index_object ).intervals( index_arg ) = varargin{ index_arg }( index_object );
                end

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                objects( index_object ).N_dimensions = numel( objects( index_object ).intervals );

            end % for index_object = 1:numel( varargin{ 1 } )

        end % function objects = orthotope( varargin )

        %------------------------------------------------------------------
        % vertices
        %------------------------------------------------------------------
        function objects_out = vertices( orthotopes )

            % specify cell array for objects_out
            objects_out = cell( size( orthotopes ) );

            % iterate orthotopes
            for index_object = 1:numel( orthotopes )

                % number of vertices
                N_dimensions_act = orthotopes( index_object ).N_dimensions;
                N_vertices_act = 2^N_dimensions_act;

                % initialize results
                objects_out{ index_object } = repmat( orthotopes( index_object ).intervals( 1 ).lb, [ N_vertices_act, N_dimensions_act ] );

                % extract vertices
                for index_vertex = 1:N_vertices_act

                    indices_bounds_act = dec2bin( index_vertex - 1, N_dimensions_act ) - '0' + 1;

                    for index_dimension = 1:N_dimensions_act

                        if indices_bounds_act( index_dimension ) == 1
                            objects_out{ index_object }( index_vertex, index_dimension ) = orthotopes( index_object ).intervals( index_dimension ).lb;
                        else
                            objects_out{ index_object }( index_vertex, index_dimension ) = orthotopes( index_object ).intervals( index_dimension ).ub;
                        end

                    end % for index_dimension = 1:N_dimensions_act

                end % for index_vertex = 1:N_vertices_act

            end % for index_object = 1:numel( orthotopes )

            % avoid cell array for single orthotope
            if isscalar( orthotopes )
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = vertices( orthotopes )

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        function objects_out = center( orthotopes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.orthotope
            if ~isa( orthotopes, 'math.orthotope' )
                errorStruct.message = 'orthotopes must be math.orthotope!';
                errorStruct.identifier = 'center:NoOrthotopes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute centers
            %--------------------------------------------------------------
            % specify cell array for objects_out
            objects_out = cell( size( orthotopes ) );

            % iterate orthotopes
            for index_object = 1:numel( orthotopes )

                % compute interval centers
                objects_out{ index_object } = center( orthotopes( index_object ).intervals );

            end % for index_object = 1:numel( orthotopes )

            % avoid cell array for single orthotope
            if isscalar( orthotopes )
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = center( orthotopes )

        %------------------------------------------------------------------
        % move orthotope
        %------------------------------------------------------------------
        function orthotopes = move( orthotopes, centers )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class math.orthotope
            if ~isa( orthotopes, 'math.orthotope' )
                errorStruct.message = 'orthotopes must be math.orthotope!';
                errorStruct.identifier = 'move:NoOrthotopes';
                error( errorStruct );
            end

            % ensure cell array for centers
            if ~iscell( centers )
                centers = { centers };
            end

            % ensure equal number of dimensions and sizes
            [ orthotopes, centers ] = auxiliary.ensureEqualSize( orthotopes, centers );

            %--------------------------------------------------------------
            % 2.) move orthotopes
            %--------------------------------------------------------------
            % iterate orthotopes
            for index_object = 1:numel( centers )

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( orthotopes( index_object ).intervals, centers{ index_object } );

                % move orthotope
                orthotopes( index_object ).intervals = move( orthotopes( index_object ).intervals, centers{ index_object } );

            end % for index_object = 1:numel( orthotopes )

        end % function orthotopes = move( orthotopes, centers )

    end % methods

end % classdef orthotope
